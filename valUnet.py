import argparse
import os
import json
import itertools
import csv
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from unet import UNet
from seg_dataset import get_patient_splits, MRISegDataset
from losses import CEDiceLoss, FocalTverskyLoss, UnifiedFocalLoss

# ==========================================
# NEW: ADDITIONAL IMPORTS FOR 3D METRICS
# ==========================================
import glob
import numpy as np
import nibabel as nib
from PIL import Image
from scipy.ndimage import binary_erosion
from scipy.spatial.distance import cdist

# Configs for 3D Metrics
SPACING_3D = (0.5, 0.5, 2.0)
CLASSES    = {1: "STN", 2: "RN"}


# ==========================================
# EVALUATION FUNCTION (Identical to original)
# ==========================================

def evaluate(model, dataloader, criterion, device, desc="Val"):
    model.eval()
    total_loss = 0.0
    stn_inter, stn_union = 0, 0
    rn_inter,  rn_union  = 0, 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc=desc, leave=False):
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


# ==========================================
# NEW: 3D METRICS AND VOLUMETRIC HELPERS
# ==========================================

def dice_score_3d(pred, gt):
    inter = np.sum(pred * gt)
    denom = np.sum(pred) + np.sum(gt)
    return float(2.0 * inter / denom) if denom > 0 else float('nan')

def precision_score_3d(pred, gt):
    tp = np.sum(pred * gt)
    fp = np.sum(pred * (1 - gt))
    return float(tp / (tp + fp)) if (tp + fp) > 0 else float('nan')

def recall_score_3d(pred, gt):
    tp = np.sum(pred * gt)
    fn = np.sum((1 - pred) * gt)
    return float(tp / (tp + fn)) if (tp + fn) > 0 else float('nan')

def iou_score_3d(pred, gt):
    inter = np.sum(pred * gt)
    union = np.sum((pred + gt) > 0)
    return float(inter / union) if union > 0 else float('nan')

def surface_voxels(mask):
    eroded = binary_erosion(mask)
    return np.argwhere(mask ^ eroded)

def hd95_3d(pred, gt, spacing=SPACING_3D):
    if np.sum(pred) == 0 or np.sum(gt) == 0:
        return float('nan')
    ps = surface_voxels(pred.astype(bool)) * np.array(spacing)
    gs = surface_voxels(gt.astype(bool)) * np.array(spacing)
    if len(ps) == 0 or len(gs) == 0:
        return float('nan')
    D = cdist(ps, gs)
    return float(np.percentile(np.concatenate([np.min(D, axis=1), np.min(D, axis=0)]), 95))

def nanmean(vals):
    finite = [v for v in vals if v is not None and np.isfinite(v)]
    return float(np.mean(finite)) if finite else float('nan')

def run_inference_to_png(model, dataset, device, pred_dir):
    os.makedirs(pred_dir, exist_ok=True)
    model.eval()
    with torch.no_grad():
        for idx in range(len(dataset)):
            data = dataset[idx]
            img_tensor = data['image'].unsqueeze(0).to(device)
            pred = torch.argmax(model(img_tensor), dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
            
            img_path, _ = dataset.samples[idx]
            orig_filename = os.path.basename(img_path)
            Image.fromarray(pred).save(os.path.join(pred_dir, orig_filename))

def reconstruct_and_score_3d(pred_dir, mask_3d_dir):
    pred_files = sorted(glob.glob(os.path.join(pred_dir, "*.png")))
    if not pred_files:
        return []
    
    subjects = sorted(set([os.path.basename(f).split("_z")[0] for f in pred_files]))
    results_3d = []

    for subj in subjects:
        gt_path = os.path.join(mask_3d_dir, f"{subj}_mask.nii.gz")
        if not os.path.exists(gt_path):
            matches = glob.glob(os.path.join(mask_3d_dir, f"{subj}*.nii.gz"))
            gt_path = matches[0] if matches else None
        if not gt_path:
            continue

        gt_nii  = nib.load(gt_path)
        gt_3d   = gt_nii.get_fdata().astype(np.uint8)
        H, W, D = gt_3d.shape

        recon_3d = np.zeros((H, W, D), dtype=np.uint8)

        for p_path in sorted(glob.glob(os.path.join(pred_dir, f"{subj}_z*.png"))):
            try:
                z_idx = int(os.path.basename(p_path).split("_z")[-1].replace(".png", ""))
                if 0 <= z_idx < D:
                    pred_slice = np.array(Image.open(p_path))
                    h_p, w_p = pred_slice.shape
                    sy, sx = (H - h_p) // 2, (W - w_p) // 2
                    recon_3d[sy:sy+h_p, sx:sx+w_p, z_idx] = pred_slice
            except Exception:
                continue

        for cls, cls_name in CLASSES.items():
            pred_bin = (recon_3d == cls).astype(np.uint8)
            gt_bin   = (gt_3d   == cls).astype(np.uint8)
            results_3d.append({
                "subject":   subj,
                "class_id":  cls,
                "class":     cls_name,
                "dice":      dice_score_3d(pred_bin, gt_bin),
                "precision": precision_score_3d(pred_bin, gt_bin),
                "recall":    recall_score_3d(pred_bin, gt_bin),
                "iou":       iou_score_3d(pred_bin, gt_bin),
                "hd95":      hd95_3d(pred_bin, gt_bin),
            })
    return results_3d


# ==========================================
# MAIN VALIDATION SCRIPT
# ==========================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser('U-Net Grid Search Post-Evaluation')
    parser.add_argument('--image_dir',  type=str, default="/home/jovyan/seg_data_modified_2d/images")
    parser.add_argument('--mask_dir',   type=str, default="/home/jovyan/seg_data_modified_2d/masks")
    parser.add_argument('--output_dir', type=str, default='./seg_grid_searchNewLosses')
    parser.add_argument('--batch_size', type=int, default=32)
    # New argument to capture ground truth original 3D NIfTI volumes location
    parser.add_argument('--mask_3d_dir', type=str, default="/home/jovyan/SegDataModified/masks_edited")
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Get data splits
    _, val_p, test_p = get_patient_splits(args.mask_dir, val_fraction=0.1, test_fraction=0.1)

    # Val and Test loaders (identical setup to your original training setup)
    val_ds   = MRISegDataset(args.image_dir, args.mask_dir, val_p,  augment=False, only_annotated=True)
    test_ds  = MRISegDataset(args.image_dir, args.mask_dir, test_p, augment=False, only_annotated=True)
    val_loader  = DataLoader(val_ds,  batch_size=args.batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # Reconstruct the hyperparameter grid
    grid_params = {
        'loss_type':       ['CEDice', 'FocalTversky', 'UnifiedFocal'],
        'features':        [[64, 128, 256, 512], [32, 64, 128, 256]],
        'lr':              [1e-4, 5e-4],
        'weight_decay':    [1e-4, 1e-2],
        'dropout':         [0.2, 0.4],
        'only_annotated':  [True],
        'loss_weights':    [(0.3, 0.7), (0.5, 0.5), (0.1, 0.9)],
    }

    keys, values = zip(*grid_params.items())
    experiments  = [dict(zip(keys, v)) for v in itertools.product(*values)]
    print(f"Total experiments to check: {len(experiments)}")

    results = []

    for config in experiments:
        ce_w, ft_w  = config['loss_weights']
        loss_name   = config['loss_type']
        only_ann    = config['only_annotated']

        # Construct folder name exactly as before
        run_name = (
            f"Unet_{loss_name}"
            f"_f{config['features'][0]}"
            f"_lr{config['lr']}"
            f"_drop{config['dropout']}"
            f"_L2{config['weight_decay']}"
            f"_ann{only_ann}"
            f"_w{ce_w}-{ft_w}"
        )
        
        run_dir = os.path.join(args.output_dir, run_name)
        best_model_path = os.path.join(run_dir, "best_model.pth")

        # Skip if this configuration was never completed/trained
        if not os.path.exists(best_model_path):
            print(f"Skipping (Checkpoint not found): {run_name}")
            continue

        print(f"Evaluating: {run_name}")

        # 1. Rebuild Model Architecture
        model = UNet(
            in_channels=1, num_classes=3,
            features=config['features'],
            dropout_rate=config['dropout'],
        ).to(device)
        
        # Load weights safely
        model.load_state_dict(torch.load(best_model_path, map_location=device))

        # 2. Rebuild Class Weights
        if only_ann:
            weights = torch.tensor([0.01,  0.5, 1.0], dtype=torch.float32).to(device)
        else:
            weights = torch.tensor([0.001, 0.5, 1.0], dtype=torch.float32).to(device)

        # 3. Rebuild Criterion
        if loss_name == 'CEDice':
            criterion = CEDiceLoss(ce_weight=ce_w, dice_weight=ft_w, class_weights=weights)
        elif loss_name == 'FocalTversky':
            criterion = FocalTverskyLoss(class_weights=weights, alpha=0.7, beta=0.3, gamma=0.75, ce_weight=ce_w, ft_weight=ft_w)
        elif loss_name == 'UnifiedFocal':
            criterion = UnifiedFocalLoss(class_weights=weights, alpha=0.7, beta=0.3, gamma=2.0)
        else:
            continue

        # 4. Evaluate Checkpoint on Validation and Test Sets
        val_loss, val_stn, val_rn = evaluate(model, val_loader, criterion, device, desc="Val Set")
        test_loss, test_stn, test_rn = evaluate(model, test_loader, criterion, device, desc="Test Set")

        # Save individual run results inside its folder without destroying anything
        with open(os.path.join(run_dir, "results_validated.json"), "w") as f:
            json.dump({
                "val_loss": val_loss,
                "val_stn_dice": val_stn,
                "val_rn_dice": val_rn,
                "test_loss": test_loss,
                "test_stn_dice": test_stn,
                "test_rn_dice": test_rn,
                "config": config
            }, f, indent=4)

        results.append({
            'config': config,
            'val_loss': val_loss,
            'val_stn_dice': val_stn,
            'val_rn_dice': val_rn,
            'test_loss': test_loss,
            'test_stn_dice': test_stn,
            'test_rn_dice': test_rn
        })

        # --------------------------------------------------
        # NEW: ADDED 3D VOLUME RECONSTRUCTION & SCORING
        # --------------------------------------------------
        val_pred_dir = os.path.join(run_dir, "predictions_val_2d")
        test_pred_dir = os.path.join(run_dir, "predictions_test_2d")
        run_inference_to_png(model, val_ds, device, val_pred_dir)
        run_inference_to_png(model, test_ds, device, test_pred_dir)

        val_3d_metrics  = reconstruct_and_score_3d(val_pred_dir, args.mask_3d_dir)
        test_3d_metrics = reconstruct_and_score_3d(test_pred_dir, args.mask_3d_dir)

        # Append computed volumetric fields cleanly to the tracking dictionary item
        current_res = results[-1]
        for split_name, split_res in [("val", val_3d_metrics), ("test", test_3d_metrics)]:
            for cls, cls_name in CLASSES.items():
                cls_res = [r for r in split_res if r['class_id'] == cls]
                for metric in ['dice', 'precision', 'recall', 'iou', 'hd95']:
                    vals = [r[metric] for r in cls_res]
                    current_res[f"{cls_name}_{split_name}_mean_{metric}"] = nanmean(vals)
                    current_res[f"{cls_name}_{split_name}_std_{metric}"]  = float(np.nanstd([v for v in vals if np.isfinite(v)])) if any(np.isfinite(v) for v in vals if v is not None) else float('nan')

        # Save complete standalone validation result including 3D metrics inside its execution path
        with open(os.path.join(run_dir, "results_validated_with_3d.json"), "w") as f:
            json.dump(current_res, f, indent=4)

    print("\n" + "="*50 + "\nEVALUATION COMPLETE - SORTING BY VALIDATION\n" + "="*50)

    # CRITICAL SORT: Sorting strictly by Validation STN Dice performance
    sorted_results = sorted(results, key=lambda x: x['val_stn_dice'], reverse=True)
    
    print("\nRanked Configurations (Best Validation STN Dice first):")
    for res in sorted_results:
        print(f"VAL STN: {res['val_stn_dice']:.4f} | VAL RN: {res['val_rn_dice']:.4f} || "
              f"TEST STN: {res['test_stn_dice']:.4f} | TEST RN: {res['test_rn_dice']:.4f} | {res['config']['loss_type']}")

    # --------------------------------------------------
    # Save brand new clean summaries
    # --------------------------------------------------
    json_path = os.path.join(args.output_dir, 'grid_search_val_summary.json')
    with open(json_path, 'w') as f:
        json.dump(sorted_results, f, indent=4)

    csv_path = os.path.join(args.output_dir, 'grid_search_val_summary.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'Val_STN_Dice', 'Val_RN_Dice', 'Val_Loss',
            'Test_STN_Dice', 'Test_RN_Dice', 'Test_Loss',
            'Loss_Type', 'Features', 'Learning_Rate',
            'Weight_Decay', 'Dropout', 'Only_Annotated',
            'CE_Weight', 'FT_Weight',
        ])
        for res in sorted_results:
            cfg = res['config']
            writer.writerow([
                f"{res['val_stn_dice']:.4f}",
                f"{res['val_rn_dice']:.4f}",
                f"{res['val_loss']:.4f}",
                f"{res['test_stn_dice']:.4f}",
                f"{res['test_rn_dice']:.4f}",
                f"{res['test_loss']:.4f}",
                cfg['loss_type'],
                str(cfg['features']),
                cfg['lr'],
                cfg['weight_decay'],
                cfg['dropout'],
                cfg['only_annotated'],
                cfg['loss_weights'][0],
                cfg['loss_weights'][1],
            ])

    print(f"\nNew validation-ranked summaries saved to:\n  {csv_path}\n  {json_path}")

    # --------------------------------------------------
    # NEW: ADDITIONAL SUMMARY FILES INCLUDING ALL 3D METRICS
    # --------------------------------------------------
    json_path_3d = os.path.join(args.output_dir, 'grid_search_3d_summary.json')
    with open(json_path_3d, 'w') as f:
        json.dump(sorted_results, f, indent=4)

    csv_path_3d = os.path.join(args.output_dir, 'grid_search_3d_summary.csv')
    if sorted_results:
        with open(csv_path_3d, 'w', newline='') as f:
            writer_3d = csv.writer(f)
            
            # Setup the complete list of 3D tracking columns (Mean & Std Dev)
            metrics_headers = []
            for split in ['val', 'test']:
                for cls in ['STN', 'RN']:
                    for m in ['dice', 'hd95', 'precision', 'recall', 'iou']:
                        metrics_headers.extend([f"{cls}_{split}_mean_{m}", f"{cls}_{split}_std_{m}"])
            
            writer_3d.writerow([
                'Val_STN_Dice_2D', 'Val_RN_Dice_2D', 'Val_Loss',
                'Test_STN_Dice_2D', 'Test_RN_Dice_2D', 'Test_Loss'
            ] + metrics_headers + [
                'Loss_Type', 'Features', 'Learning_Rate',
                'Weight_Decay', 'Dropout', 'Only_Annotated',
                'CE_Weight', 'FT_Weight',
            ])
            
            for res in sorted_results:
                cfg = res['config']
                metrics_values = [f"{res.get(h, float('nan')):.4f}" if 'hd95' not in h else f"{res.get(h, float('nan')):.2f}" for h in metrics_headers]
                writer_3d.writerow([
                    f"{res['val_stn_dice']:.4f}",
                    f"{res['val_rn_dice']:.4f}",
                    f"{res['val_loss']:.4f}",
                    f"{res['test_stn_dice']:.4f}",
                    f"{res['test_rn_dice']:.4f}",
                    f"{res['test_loss']:.4f}"
                ] + metrics_values + [
                    cfg['loss_type'],
                    str(cfg['features']),
                    cfg['lr'],
                    cfg['weight_decay'],
                    cfg['dropout'],
                    cfg['only_annotated'],
                    cfg['loss_weights'][0],
                    cfg['loss_weights'][1],
                ])

    print(f"New volume-inclusive 3D metrics summaries saved to:\n  {csv_path_3d}\n  {json_path_3d}")