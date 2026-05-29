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
# EARLY STOPPING
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


# ==========================================
# EVALUATION
# Lightweight Dice-only evaluation used during
# grid search to keep memory cost low.
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
# SINGLE EXPERIMENT
# ==========================================

def train_experiment(config, train_loader, val_loader, test_loader, args, device):
    """Run one grid-search configuration and return test Dice scores."""
    ce_w, ft_w  = config['loss_weights']
    loss_name   = config['loss_type']
    only_ann    = config['only_annotated']

    run_name = (
        f"Unet_{loss_name}"
        f"_f{config['features'][0]}"
        f"_lr{config['lr']}"
        f"_drop{config['dropout']}"
        f"_L2{config['weight_decay']}"
        f"_ann{only_ann}"
        f"_w{ce_w}-{ft_w}"
    )
    print(f"\n{'='*60}\nGrid Search Run: {run_name}\n{'='*60}")

    run_dir         = os.path.join(args.output_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)
    best_model_path = os.path.join(run_dir, "best_model.pth")
    log_path        = os.path.join(run_dir, "train.log")

    with open(log_path, "w") as f:
        f.write(f"--- Config: {config} ---\n\n")

    # --------------------------------------------------
    # Model
    # --------------------------------------------------
    model = UNet(
        in_channels=1, num_classes=3,
        features=config['features'],
        dropout_rate=config['dropout'],
    ).to(device)

    # --------------------------------------------------
    # Class weights (thesis Section 3.4)
    #   Annotated only:  w = [0.01,  0.5, 1.0]
    #   All slices:      w = [0.001, 0.5, 1.0]
    # --------------------------------------------------
    if only_ann:
        weights = torch.tensor([0.01,  0.5, 1.0], dtype=torch.float32).to(device)
    else:
        weights = torch.tensor([0.001, 0.5, 1.0], dtype=torch.float32).to(device)

    # --------------------------------------------------
    # Loss
    # --------------------------------------------------
    if loss_name == 'CEDice':
      
        criterion = CEDiceLoss(
            ce_weight=ce_w, dice_weight=ft_w,
            class_weights=weights,
        )
    elif loss_name == 'FocalTversky':
     
        criterion = FocalTverskyLoss(
            class_weights=weights,
            alpha=0.7, beta=0.3, gamma=0.75,
            ce_weight=ce_w, ft_weight=ft_w,
        )
    elif loss_name == 'UnifiedFocal':

        criterion = UnifiedFocalLoss(
            class_weights=weights,
            alpha=0.7, beta=0.3, gamma=2.0,
        )
    else:
        raise ValueError(f"Unknown loss_type: {loss_name}")

    # --------------------------------------------------
    # Optimiser + early stopping
    # --------------------------------------------------
    optimizer     = torch.optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    early_stop    = EarlyStopping(patience=15, verbose=True)
    history       = {'train_loss': [], 'val_loss': [], 'val_dice_stn': [], 'val_dice_rn': []}

    # --------------------------------------------------
    # Training loop
    # --------------------------------------------------
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}", leave=False):
            images, labels = batch['image'].to(device), batch['label'].to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)
        val_loss, dice_stn, dice_rn = evaluate(
            model, val_loader, criterion, device, desc=f"Epoch {epoch+1} Val"
        )

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_dice_stn'].append(dice_stn)
        history['val_dice_rn'].append(dice_rn)

        log_line = (
            f"Epoch {epoch+1:03d} | Train: {train_loss:.4f} | "
            f"Val: {val_loss:.4f} | STN: {dice_stn:.4f} | RN: {dice_rn:.4f}"
        )
        print(log_line)
        with open(log_path, "a") as f:
            f.write(log_line + "\n")

        early_stop(val_loss, model, best_model_path)
        if early_stop.early_stop:
            print("Early stopping triggered.")
            break

    # --------------------------------------------------
    # Test evaluation on best checkpoint
    # --------------------------------------------------
    print("\nEvaluating best model on Test Set...")
    model.load_state_dict(torch.load(best_model_path))
    test_loss, test_stn, test_rn = evaluate(model, test_loader, criterion, device, desc="Test")

    test_log = (
        f"\nTest -> Loss: {test_loss:.4f} | STN Dice: {test_stn:.4f} | RN Dice: {test_rn:.4f}\n"
    )
    print(test_log.strip())
    with open(log_path, "a") as f:
        f.write(test_log)

    with open(os.path.join(run_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=4)

    with open(os.path.join(run_dir, "results.json"), "w") as f:
        json.dump({
            "test_loss":     test_loss,
            "test_stn_dice": test_stn,
            "test_rn_dice":  test_rn,
            "config":        config,
        }, f, indent=4)

    return test_stn, test_rn


# ==========================================
# ENTRY POINT
# ==========================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser('U-Net Grid Search')
    parser.add_argument('--image_dir',  type=str, default="/home/jovyan/seg_data_modified_2d/images")
    parser.add_argument('--mask_dir',   type=str, default="/home/jovyan/seg_data_modified_2d/masks")
    parser.add_argument('--output_dir', type=str, default='./seg_grid_searchNewLosses')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs',     type=int, default=200)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.output_dir, exist_ok=True)

    train_p, val_p, test_p = get_patient_splits(
        args.mask_dir, val_fraction=0.1, test_fraction=0.1
    )

    # Val and Test always use annotated slices only for fair evaluation
    val_ds   = MRISegDataset(args.image_dir, args.mask_dir, val_p,  augment=False, only_annotated=True)
    test_ds  = MRISegDataset(args.image_dir, args.mask_dir, test_p, augment=False, only_annotated=True)
    val_loader  = DataLoader(val_ds,  batch_size=args.batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # --------------------------------------------------
    # Grid search hyperparameters
    # --------------------------------------------------
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
    print(f"Total experiments: {len(experiments)}")

    results = []
    for config in experiments:
        train_ds = MRISegDataset(
            args.image_dir, args.mask_dir, train_p,
            augment=True, only_annotated=config['only_annotated'],
        )
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4)

        stn_score, rn_score = train_experiment(
            config, train_loader, val_loader, test_loader, args, device
        )
        results.append({'config': config, 'stn_dice': stn_score, 'rn_dice': rn_score})

    print("\n" + "="*50 + "\nGRID SEARCH COMPLETE\n" + "="*50)

    sorted_results = sorted(results, key=lambda x: x['stn_dice'], reverse=True)
    for res in sorted_results:
        print(f"STN: {res['stn_dice']:.4f} | RN: {res['rn_dice']:.4f} | {res['config']}")

    # --------------------------------------------------
    # Save summary
    # --------------------------------------------------
    json_path = os.path.join(args.output_dir, 'grid_search_summary.json')
    with open(json_path, 'w') as f:
        json.dump(sorted_results, f, indent=4)

    csv_path = os.path.join(args.output_dir, 'grid_search_summary.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'STN_Test_Dice', 'RN_Test_Dice',
            'Loss_Type', 'Features', 'Learning_Rate',
            'Weight_Decay', 'Dropout', 'Only_Annotated',
            'CE_Weight', 'FT_Weight',
        ])
        for res in sorted_results:
            cfg = res['config']
            writer.writerow([
                f"{res['stn_dice']:.4f}",
                f"{res['rn_dice']:.4f}",
                cfg['loss_type'],
                str(cfg['features']),
                cfg['lr'],
                cfg['weight_decay'],
                cfg['dropout'],
                cfg['only_annotated'],
                cfg['loss_weights'][0],
                cfg['loss_weights'][1],
            ])

    print(f"\nSummary saved to:\n  {csv_path}\n  {json_path}")