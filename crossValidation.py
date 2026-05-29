import argparse
import os
import re
import json
import csv
import torch
import numpy as np
import time
import math
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.model_selection import ShuffleSplit

from unet import UNet
from seg_dataset import MRISegDataset
from losses import FocalTverskyLoss

'''
Cross validation zas deployed only for best models
Cross validation was preformed using ShuffleSplit to have the same validation size as for the grid search
'''

# ==========================================
# EARLY STOPPING & EVALUATION
# ==========================================
class EarlyStopping:
    """Stop training if validation loss does not improve by delta for patience epochs."""
    def __init__(self, patience=15, verbose=False, delta=0.001):
        self.patience     = patience
        self.verbose      = verbose
        self.counter      = 0
        self.best_score   = None
        self.early_stop   = False
        self.val_loss_min = float('inf')
        self.delta        = delta

    def __call__(self, val_loss, model, path):
        # Prevent NaN results from overwriting valid check-pointing records
        if math.isnan(val_loss):
            if self.verbose:
                print("EarlyStopping: Validation loss is NaN. Skipping checkpoint updates to save best model weights.")
            return

        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self._save(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self._save(val_loss, model, path)
            self.counter = 0

    def _save(self, val_loss, model, path):
        if self.verbose:
            print(f"Val loss improved ({self.val_loss_min:.6f} -> {val_loss:.6f}). Saving.")
        torch.save(model.state_dict(), path)
        self.val_loss_min = val_loss


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
# SINGLE FOLD TRAINING
# ==========================================
def train_fold(config, train_loader, val_loader, test_loader, fold, args, device, base_run_dir):
    ce_w, ft_w = config['loss_weights']
    
    run_dir = os.path.join(base_run_dir, f"fold_{fold+1}")
    os.makedirs(run_dir, exist_ok=True)
    best_model_path = os.path.join(run_dir, "best_model.pth")
    log_path = os.path.join(run_dir, "train.log")
    history_path = os.path.join(run_dir, "history.json")

    model = UNet(
        in_channels=1, num_classes=3,
        features=config['features'],
        dropout_rate=config['dropout'],
    ).to(device)

    if config['only_annotated']:
        weights = torch.tensor([0.01,  0.5, 1.0], dtype=torch.float32).to(device)
    else:
        weights = torch.tensor([0.001, 0.5, 1.0], dtype=torch.float32).to(device)

    criterion = FocalTverskyLoss(
        class_weights=weights,
        alpha=0.7, beta=0.3, gamma=0.75,
        ce_weight=ce_w, ft_weight=ft_w,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    early_stop = EarlyStopping(patience=40, verbose=True)
    
    history = []

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        nan_detected = False
        
        stn_inter, stn_union = 0, 0
        rn_inter, rn_union = 0, 0

        pbar = tqdm(train_loader, desc=f"Fold {fold+1} - Epoch {epoch+1}/{args.epochs}", leave=False)
        
        # Time tracking
        start_train_time = time.perf_counter()
        
        for batch in pbar:
            images, labels = batch['image'].to(device), batch['label'].to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            if torch.isnan(loss):
                print(f"\nNaN detected in training loss at Epoch {epoch+1}. Halting optimization.")
                nan_detected = True
                break

            loss.backward()
            

            if config['clip_gradients']:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
            optimizer.step()
            train_loss += loss.item()

            with torch.no_grad():
                preds = torch.argmax(outputs, dim=1)
                p_stn, t_stn = (preds == 1), (labels == 1)
                stn_inter += (p_stn & t_stn).sum().item()
                stn_union += (p_stn.sum() + t_stn.sum()).item()

                p_rn, t_rn = (preds == 2), (labels == 2)
                rn_inter += (p_rn & t_rn).sum().item()
                rn_union += (p_rn.sum() + t_rn.sum()).item()
                
                cur_stn = (2.0 * stn_inter) / (stn_union + 1e-6) if stn_union > 0 else 0
                cur_rn  = (2.0 * rn_inter)  / (rn_union + 1e-6)  if rn_union > 0 else 0
                pbar.set_postfix(Loss=loss.item(), STN=f"{cur_stn:.4f}", RN=f"{cur_rn:.4f}")

        if nan_detected:
            break

        epoch_train_elapsed = time.perf_counter() - start_train_time

        train_loss /= len(train_loader)
        train_dice_stn = (2.0 * stn_inter) / (stn_union + 1e-6) if stn_union > 0 else float('nan')
        train_dice_rn  = (2.0 * rn_inter)  / (rn_union  + 1e-6) if rn_union  > 0 else float('nan')
        
        # ⏱️ TIME TRACKING: Start Validation Timer
        start_val_time = time.perf_counter()
        val_loss, val_dice_stn, val_dice_rn = evaluate(model, val_loader, criterion, device, desc=f"Epoch {epoch+1} Val")
        epoch_val_elapsed = time.perf_counter() - start_val_time

        if math.isnan(val_loss):
            print(f"\n Validation evaluation returned NaN at Epoch {epoch+1}. Breaking execution sequence.")
            break

        log_line = (f"Epoch {epoch+1:03d} | Train Loss: {train_loss:.4f} | Train STN: {train_dice_stn:.4f} | Train RN: {train_dice_rn:.4f} || "
                    f"Val Loss: {val_loss:.4f} | Val STN: {val_dice_stn:.4f} | Val RN: {val_dice_rn:.4f} | "
                    f"Train Time: {epoch_train_elapsed:.2f}s | Val Time: {epoch_val_elapsed:.2f}s")
        with open(log_path, "a") as f:
            f.write(log_line + "\n")

        history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss, "train_stn": train_dice_stn, "train_rn": train_dice_rn,
            "val_loss": val_loss, "val_stn": val_dice_stn, "val_rn": val_dice_rn,
            "train_time_sec": epoch_train_elapsed, "val_time_sec": epoch_val_elapsed
        })

        with open(history_path, "w") as f:
            json.dump(history, f, indent=4)

        early_stop(val_loss, model, best_model_path)
        if early_stop.early_stop:
            break

    # ⏱️ Time tracking
    print(f"=> Loading uncorrupted best checkpoint target from: {best_model_path}")
    model.load_state_dict(torch.load(best_model_path))
    
    start_test_time = time.perf_counter()
    test_loss, test_stn, test_rn = evaluate(model, test_loader, criterion, device, desc=f"Fold {fold+1} Test Eval")
    test_elapsed_time = time.perf_counter() - start_test_time
    
    test_log_line = f"Fold {fold+1} Test Results -> STN Dice: {test_stn:.4f} | RN Dice: {test_rn:.4f} | Test Inference Time: {test_elapsed_time:.4f}s"
    print("\n" + test_log_line)
    with open(log_path, "a") as f:
        f.write("\n" + test_log_line + "\n")

    return test_stn, test_rn, history, test_elapsed_time


# ==========================================
# PATIENT GATHERING HELPER
# ==========================================
def get_all_annotated_patients(mask_dir):
    print("Scanning mask directory to find annotated patients...")
    annotated_patients = set()
    for fn in os.listdir(mask_dir):
        if not fn.endswith('.png'):
            continue
        mask = np.array(Image.open(os.path.join(mask_dir, fn)))
        if mask.max() > 0:
            name = os.path.splitext(fn)[0]
            pid = re.split(r'_z\d+', name)[0]
            annotated_patients.add(pid)
    
    patients = list(annotated_patients)
    patients.sort()
    return patients


# ==========================================
# ENTRY POINT
# ==========================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser('U-Net 5-Fold CV Fixed Test - Double Scenario Speed Benchmarked')
    parser.add_argument('--image_dir',  type=str, default="/home/jovyan/seg_data_modified_2d/images")
    parser.add_argument('--mask_dir',   type=str, default="/home/jovyan/seg_data_modified_2d/masks")
    parser.add_argument('--output_dir', type=str, default='./seg_cv_results')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs',     type=int, default=200)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.output_dir, exist_ok=True)

    #  Dynamically generate 8 configurations (Combinations of architectures, annotations, and gradient clipping)
    cv_configs = []
    for ann in [True, False]:
        for clip in [True, False]:
            clip_str = "clipTrue" if clip else "clipFalse"
            cv_configs.append({
                'name': f"Config1_f32_drop0.2_L20.0001_ann{ann}_{clip_str}",
                'loss_type': 'FocalTversky',
                'features': [32, 64, 128, 256],
                'lr': 0.0005,
                'dropout': 0.2,
                'weight_decay': 0.0001,
                'only_annotated': ann,
                'clip_gradients': clip,
                'loss_weights': (0.1, 0.9)
            })
            cv_configs.append({
                'name': f"Config2_f64_drop0.4_L20.01_ann{ann}_{clip_str}",
                'loss_type': 'FocalTversky',
                'features': [64, 128, 256, 512],
                'lr': 0.0005,
                'dropout': 0.4,
                'weight_decay': 0.01,
                'only_annotated': ann,
                'clip_gradients': clip,
                'loss_weights': (0.1, 0.9)
            })

    FIXED_TEST_PATIENTS = ['P003', 'P011', 'P023_1', 'P046']
    
    all_patients = get_all_annotated_patients(args.mask_dir)
    cv_pool = [p for p in all_patients if p not in FIXED_TEST_PATIENTS]
    cv_pool = np.array(cv_pool)
    
    print(f"Total Annotated Patients: {len(all_patients)}")
    print(f"Fixed Test Set: {FIXED_TEST_PATIENTS}")
    print(f"Patients in CV Pool (Train/Val): {len(cv_pool)}")
    
    ss = ShuffleSplit(n_splits=5, train_size=38, test_size=4, random_state=42)
    cv_summary = []

    for config in cv_configs:
        print(f"\n{'='*70}\nRunning 5-Fold CV: {config['name']}\n{'='*70}")
        
        base_run_dir = os.path.join(args.output_dir, config['name'])
        os.makedirs(base_run_dir, exist_ok=True)
        
        fold_stn_scores = []
        fold_rn_scores = []
        
        # Lists to gather speed performance profiles across independent iterations
        fold_avg_train_times = []
        fold_avg_val_times = []
        fold_test_inference_times = []

        test_ds = MRISegDataset(
            args.image_dir, args.mask_dir, FIXED_TEST_PATIENTS, 
            augment=False, only_annotated=config['only_annotated']
        )
        test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)

        for fold, (train_idx, val_idx) in enumerate(ss.split(cv_pool)):
            run_dir = os.path.join(base_run_dir, f"fold_{fold+1}")
            history_path = os.path.join(run_dir, "history.json")
            best_model_path = os.path.join(run_dir, "best_model.pth")
            
            # Conditional processing path check
            needs_retraining = True
            if os.path.exists(history_path) and os.path.exists(best_model_path):
                try:
                    with open(history_path, "r") as f:
                        history_data = json.load(f)
                    has_nan = any(
                        math.isnan(h.get("train_loss", 0)) or math.isnan(h.get("val_loss", 0))
                        for h in history_data
                    )
                    if not has_nan and len(history_data) > 0:
                        needs_retraining = False
                except Exception:
                    needs_retraining = True

            train_p = cv_pool[train_idx].tolist()
            val_p   = cv_pool[val_idx].tolist()

            train_ds = MRISegDataset(args.image_dir, args.mask_dir, train_p, augment=True, only_annotated=config['only_annotated'])
            val_ds = MRISegDataset(args.image_dir, args.mask_dir, val_p, augment=False, only_annotated=config['only_annotated'])

            train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  num_workers=4)
            val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=4)

            if needs_retraining:
                print(f"\n⚡ Executing Fold {fold+1}/5 Training Block...")
                log_file = os.path.join(run_dir, "train.log")
                if os.path.exists(log_file):
                    os.remove(log_file)

                test_stn_score, test_rn_score, fold_history, test_elapsed = train_fold(
                    config, train_loader, val_loader, test_loader, fold, args, device, base_run_dir
                )
            else:
                print(f"Fold {fold+1}/5 Healthy & Validated. Fetching performance telemetry directly.")
                model = UNet(in_channels=1, num_classes=3, features=config['features'], dropout_rate=config['dropout']).to(device)
                model.load_state_dict(torch.load(best_model_path, map_location=device))
                
                ce_w, ft_w = config['loss_weights']
                weights = torch.tensor([0.01, 0.5, 1.0] if config['only_annotated'] else [0.001, 0.5, 1.0], dtype=torch.float32).to(device)
                criterion = FocalTverskyLoss(class_weights=weights, alpha=0.7, beta=0.3, gamma=0.75, ce_weight=ce_w, ft_weight=ft_w)
                
                start_test_time = time.perf_counter()
                _, test_stn_score, test_rn_score = evaluate(model, test_loader, criterion, device, desc=f"Fold {fold+1} Quick-Eval")
                test_elapsed = time.perf_counter() - start_test_time
                with open(history_path, "r") as f:
                    fold_history = json.load(f)

            fold_stn_scores.append(test_stn_score)
            fold_rn_scores.append(test_rn_score)
            
            # Post-process operational speed tracking records for compilation
            t_times = [h["train_time_sec"] for h in fold_history if "train_time_sec" in h]
            v_times = [h["val_time_sec"] for h in fold_history if "val_time_sec" in h]
            
            fold_avg_train_times.append(np.mean(t_times) if t_times else 0.0)
            fold_avg_val_times.append(np.mean(v_times) if v_times else 0.0)
            fold_test_inference_times.append(test_elapsed)

        mean_stn = np.mean(fold_stn_scores)
        std_stn  = np.std(fold_stn_scores)
        mean_rn  = np.mean(fold_rn_scores)
        std_rn   = np.std(fold_rn_scores)

        print(f"\n{'*'*50}")
        print(f"Final FIXED TEST CV Summary for {config['name']}")
        print(f"STN Dice: {mean_stn:.4f} ± {std_stn:.4f}")
        print(f"RN Dice:  {mean_rn:.4f}  ± {std_rn:.4f}")
        print(f"Avg Train Time/Epoch: {np.mean(fold_avg_train_times):.2f}s")
        print(f"Avg Val Time/Epoch:   {np.mean(fold_avg_val_times):.2f}s")
        print(f"Avg Inference Time:   {np.mean(fold_test_inference_times):.4f}s")
        print(f"{'*'*50}\n")

        cv_summary.append({
            'config_name': config['name'],
            'mean_test_stn': mean_stn, 'std_test_stn': std_stn,
            'mean_test_rn': mean_rn,   'std_test_rn': std_rn,
            'avg_epoch_train_time_sec': float(np.mean(fold_avg_train_times)),
            'avg_epoch_val_time_sec': float(np.mean(fold_avg_val_times)),
            'avg_test_inference_time_sec': float(np.mean(fold_test_inference_times)),
            'folds_test_stn': fold_stn_scores,
            'folds_test_rn': fold_rn_scores
        })

    # ==========================================
    # SAVE GLOBAL EXCEL/JSON METRICS REPORT
    # ==========================================
    json_path = os.path.join(args.output_dir, 'results.json')
    with open(json_path, 'w') as f:
        json.dump(cv_summary, f, indent=4)

    csv_path = os.path.join(args.output_dir, 'results.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'Config_Name', 'Mean_Test_STN', 'Std_Test_STN', 'Mean_Test_RN', 'Std_Test_RN',
            'Avg_Train_Time_Per_Epoch(s)', 'Avg_Val_Time_Per_Epoch(s)', 'Avg_Test_Inference_Time(s)'
        ])
        for res in cv_summary:
            writer.writerow([
                res['config_name'],
                f"{res['mean_test_stn']:.4f}", f"{res['std_test_stn']:.4f}",
                f"{res['mean_test_rn']:.4f}",  f"{res['std_test_rn']:.4f}",
                f"{res['avg_epoch_train_time_sec']:.2f}",
                f"{res['avg_epoch_val_time_sec']:.2f}",
                f"{res['avg_test_inference_time_sec']:.4f}"
            ])

    print(f"\nCross Validation operations successful. Consolidated diagnostic tables saved to: {args.output_dir}")