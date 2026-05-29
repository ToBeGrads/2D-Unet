import argparse
import os
import re
import json
import csv
import torch
import numpy as np
import time
from tqdm import tqdm
from PIL import Image
from torch.utils.data import DataLoader
from sklearn.model_selection import ShuffleSplit

from unet import UNet
from seg_dataset import MRISegDataset
from losses import FocalTverskyLoss

# ==========================================
# EVALUATION FUNCTION (LOSS & DICE ONLY)
# ==========================================
def benchmark_evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    stn_inter, stn_union = 0, 0
    rn_inter,  rn_union  = 0, 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Benchmarking", leave=False):
            images = batch['image'].to(device)
            labels = batch['label'].to(device)
            outputs = model(images)
            
            total_loss += criterion(outputs, labels).item()
            preds = torch.argmax(outputs, dim=1)

            p_stn, t_stn = (preds == 1), (labels == 1)
            stn_inter += (p_stn & t_stn).sum().item()
            stn_union += (p_stn.sum() + t_stn.sum()).item()

            p_rn, t_rn = (preds == 2), (labels == 2)
            rn_inter += (p_rn & t_rn).sum().item()
            rn_union += (p_rn.sum() + t_rn.sum()).item()

    avg_loss = total_loss / len(dataloader)
    dice_stn = (2.0 * stn_inter) / (stn_union + 1e-6) if stn_union > 0 else float('nan')
    dice_rn  = (2.0 * rn_inter)  / (rn_union  + 1e-6) if rn_union  > 0 else float('nan')
    return avg_loss, dice_stn, dice_rn


def get_all_annotated_patients(mask_dir):
    annotated_patients = set()
    for fn in os.listdir(mask_dir):
        if not fn.endswith('.png'): continue
        mask = np.array(Image.open(os.path.join(mask_dir, fn)))
        if mask.max() > 0:
            name = os.path.splitext(fn)[0]
            pid = re.split(r'_z\d+', name)[0]
            annotated_patients.add(pid)
    patients = list(annotated_patients)
    patients.sort()
    return patients


if __name__ == '__main__':
    parser = argparse.ArgumentParser('U-Net Validation Set Standalone Benchmark Tool')
    parser.add_argument('--image_dir',  type=str, default="/home/jovyan/seg_data_modified_2d/images")
    parser.add_argument('--mask_dir',   type=str, default="/home/jovyan/seg_data_modified_2d/masks")
    parser.add_argument('--output_dir', type=str, default='./seg_cv_results')
    parser.add_argument('--batch_size', type=int, default=32)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    cv_configs = []
    for ann in [True, False]:
        for clip in [True, False]:
            clip_str = "clipTrue" if clip else "clipFalse"
            cv_configs.append({
                'name': f"Config1_f32_drop0.2_L20.0001_ann{ann}_{clip_str}",
                'features': [32, 64, 128, 256], 'dropout': 0.2, 'only_annotated': ann, 'loss_weights': (0.1, 0.9)
            })
            cv_configs.append({
                'name': f"Config2_f64_drop0.4_L20.01_ann{ann}_{clip_str}",
                'features': [64, 128, 256, 512], 'dropout': 0.4, 'only_annotated': ann, 'loss_weights': (0.1, 0.9)
            })

    FIXED_TEST_PATIENTS = ['P003', 'P011', 'P023_1', 'P046']
    all_patients = get_all_annotated_patients(args.mask_dir)
    cv_pool = np.array([p for p in all_patients if p not in FIXED_TEST_PATIENTS])
    
    ss = ShuffleSplit(n_splits=5, train_size=38, test_size=4, random_state=42)
    benchmark_summary = []

    print(f"\n🚀 Beginning validation-set benchmarking across architectures...")

    for config in cv_configs:
        base_run_dir = os.path.join(args.output_dir, config['name'])
        if not os.path.exists(base_run_dir):
            continue

        print(f"\n=== Benchmarking Config: {config['name']} ===")

        for fold, (train_idx, val_idx) in enumerate(ss.split(cv_pool)):
            run_dir = os.path.join(base_run_dir, f"fold_{fold+1}")
            best_model_path = os.path.join(run_dir, "best_model.pth")

            if not os.path.exists(best_model_path):
                print(f" Fold {fold+1} model weights not found. Skipping.")
                continue

            val_p = cv_pool[val_idx].tolist()
            val_ds = MRISegDataset(args.image_dir, args.mask_dir, val_p, augment=False, only_annotated=config['only_annotated'])
            val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)

            model = UNet(in_channels=1, num_classes=3, features=config['features'], dropout_rate=config['dropout']).to(device)
            model.load_state_dict(torch.load(best_model_path, map_location=device))

            ce_w, ft_w = config['loss_weights']
            weights = torch.tensor([0.01, 0.5, 1.0] if config['only_annotated'] else [0.001, 0.5, 1.0], dtype=torch.float32).to(device)
            criterion = FocalTverskyLoss(class_weights=weights, alpha=0.7, beta=0.3, gamma=0.75, ce_weight=ce_w, ft_weight=ft_w)

            print(f"  ⚡ Running Evaluation on Fold {fold+1} Validation Set...")
            start_time = time.perf_counter()
            val_loss, dice_stn, dice_rn = benchmark_evaluate(model, val_loader, criterion, device)
            elapsed = time.perf_counter() - start_time

            print(f"    Fold {fold+1} -> Loss: {val_loss:.4f} | STN Dice: {dice_stn:.4f} | RN Dice: {dice_rn:.4f}")

            benchmark_summary.append({
                'Config_Name': config['name'], 'Fold': fold + 1,
                'Val_Loss': val_loss, 'Val_STN_Dice': dice_stn, 'Val_RN_Dice': dice_rn,
                'Eval_Time_Sec': elapsed
            })

    # Save outputs
    csv_out_path = os.path.join(args.output_dir, 'validation_benchmark_results.csv')
    with open(csv_out_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Config_Name', 'Fold', 'Val_Loss', 'Val_STN_Dice', 'Val_RN_Dice', 'Eval_Time_Sec'])
        for row in benchmark_summary:
            writer.writerow([row['Config_Name'], row['Fold'], f"{row['Val_Loss']:.4f}", f"{row['Val_STN_Dice']:.4f}", f"{row['Val_RN_Dice']:.4f}", f"{row['Eval_Time_Sec']:.2f}"])

    print(f"\n📊 Complete! Results saved to: {csv_out_path}")