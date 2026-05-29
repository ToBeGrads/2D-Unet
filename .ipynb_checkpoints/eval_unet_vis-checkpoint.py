import argparse
import os
import re
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import distance_transform_edt
from tqdm import tqdm

from unet import UNet
from seg_dataset import get_patient_splits, MRISegDataset

def compute_metrics(pred, gt, class_idx, spacing=(0.5, 0.5)):
    """Calculate Dice, HD95, Precision, Recall for a specific class."""
    p = (pred == class_idx)
    t = (gt == class_idx)
    
    tp = (p & t).sum()
    fp = (p & ~t).sum()
    fn = (~p & t).sum()
    tn = (~p & ~t).sum()
    
    union = p.sum() + t.sum()
    dice = float(2. * tp / union) if union > 0 else float('nan')
    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else float('nan')
    recall = float(tp / (tp + fn)) if (tp + fn) > 0 else float('nan')
    
    if np.sum(p) == 0 or np.sum(t) == 0:
        hd95 = float('nan')
    else:
        dt_gt = distance_transform_edt(~t.astype(bool), sampling=spacing)
        dt_pred = distance_transform_edt(~p.astype(bool), sampling=spacing)
        dist_p_to_g = dt_gt[p]
        dist_g_to_p = dt_pred[t]
        hd95 = float(np.percentile(np.concatenate([dist_p_to_g, dist_g_to_p]), 95))
        
    return {"dice": dice, "hd95": hd95, "precision": precision, "recall": recall}

def mask_to_rgb(mask):
    h, w = mask.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    rgb[mask == 1] = [0, 150, 255]  # STN -> Blue
    rgb[mask == 2] = [255, 50, 50]  # RN  -> Red
    return rgb

def get_patient_slice_info(dataset, idx):
    """Extract Patient ID and Slice number exactly matching seg_dataset.py logic."""
    img_path, _ = dataset.samples[idx]
    filename = os.path.basename(img_path)
    name = os.path.splitext(filename)[0]
    
    # Patient ID: P001_z00 -> P001 | P012_1_z05 -> P012_1
    patient_id = re.split(r'_z\d+', name)[0]
    
    # Slice Number: P001_z00 -> z00
    match = re.search(r'_z(\d+)', name)
    slice_num = "z" + match.group(1) if match else "Unknown"
    
    return patient_id, slice_num

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. Load Test Split
    _, _, test_p = get_patient_splits(args.mask_dir, val_fraction=0.1, test_fraction=0.1)
    test_dataset = MRISegDataset(args.image_dir, args.mask_dir, test_p, augment=False, only_annotated=True)
    
    total_slices = len(test_dataset)
    print(f"✅ Found {total_slices} annotated test slices in the test split.")
    if total_slices == 0:
        print("⚠️  No annotated slices found. Check paths or try only_annotated=False.")
        return

    # 2. Initialize & Load Model
    model = UNet(in_channels=1, num_classes=3, features=[64, 128, 256, 512])
    model.load_state_dict(torch.load(args.weights, map_location=device))
    model.to(device).eval()

    # 3. Output Directory
    out_dir = os.path.join(os.path.dirname(args.weights), "unet_visualizations")
    os.makedirs(out_dir, exist_ok=True)

    # 4. Containers for metrics
    all_metrics = {
        1: {"dice": [], "hd95": [], "precision": [], "recall": []},  # STN
        2: {"dice": [], "hd95": [], "precision": [], "recall": []}   # RN
    }

    print(f"\n🔄 Processing all {total_slices} slices...\n")
    for idx in range(total_slices):
        data = test_dataset[idx]
        img_tensor = data['image'].unsqueeze(0).to(device)
        gt = data['label'].numpy()
        
        with torch.no_grad():
            pred = torch.argmax(model(img_tensor), dim=1).squeeze(0).cpu().numpy()
            
        m_stn = compute_metrics(pred, gt, 1)
        m_rn  = compute_metrics(pred, gt, 2)
        
        # Store metrics for final averaging
        for cid, m in [(1, m_stn), (2, m_rn)]:
            for k in ["dice", "hd95", "precision", "recall"]:
                all_metrics[cid][k].append(m[k])

        patient_id, slice_num = get_patient_slice_info(test_dataset, idx)
        
        # Print per-slice metrics (compact format)
        stn_d = f"{m_stn['dice']:.3f}" if not np.isnan(m_stn['dice']) else "N/A"
        rn_d  = f"{m_rn['dice']:.3f}" if not np.isnan(m_rn['dice']) else "N/A"
        print(f"[{idx+1:02d}/{total_slices}] {patient_id:10s} {slice_num:5s} | STN Dice: {stn_d} | RN Dice: {rn_d}")
        
        # 🔄 Rotate 90° LEFT (Counter-Clockwise)
        mri_rot   = np.rot90(img_tensor.squeeze().cpu().numpy(), k=-1)
        gt_rot    = np.rot90(gt, k=-1)
        pred_rot  = np.rot90(pred, k=-1)
        
        gt_rgb   = mask_to_rgb(gt_rot)
        pred_rgb = mask_to_rgb(pred_rot)
        
        # Format metrics string for plot
        def fmt(m):
            d = f"{m['dice']:.3f}" if not np.isnan(m['dice']) else "N/A"
            h = f"{m['hd95']:.3f}" if not np.isnan(m['hd95']) else "N/A"
            p = f"{m['precision']:.3f}" if not np.isnan(m['precision']) else "N/A"
            r = f"{m['recall']:.3f}" if not np.isnan(m['recall']) else "N/A"
            return f"Dice={d} | HD95={h} | Prec={p} | Rec={r}"
            
        stn_txt = f"STN: {fmt(m_stn)}"
        rn_txt  = f"RN:  {fmt(m_rn)}"
        
        # Plot
        fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
        fig.suptitle(f"Patient: {patient_id}  |  Slice: {slice_num}", fontsize=13, fontweight='bold', y=1.05)
        
        axes[0].imshow(mri_rot, cmap='gray')
        axes[0].set_title("MRI Input")
        axes[0].axis('off')
        
        axes[1].imshow(mri_rot, cmap='gray')
        axes[1].imshow(gt_rgb, alpha=0.45)
        axes[1].set_title("Ground Truth")
        axes[1].axis('off')
        
        axes[2].imshow(mri_rot, cmap='gray')
        axes[2].imshow(pred_rgb, alpha=0.45)
        axes[2].set_title(f"Prediction\n{stn_txt}\n{rn_txt}", fontsize=10)
        axes[2].axis('off')
        
        plt.tight_layout()
        save_path = os.path.join(out_dir, f"UNet_{patient_id}_{slice_num}.png")
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        plt.close(fig) # Frees memory
        
    # 5. Calculate & Print Final Averages
    print("\n" + "="*70)
    print("FINAL TEST SET MEAN METRICS")
    print("="*70)
    for cid, name in [(1, "STN"), (2, "RN")]:
        dices = all_metrics[cid]["dice"]
        hd95s = all_metrics[cid]["hd95"]
        precs = all_metrics[cid]["precision"]
        recs  = all_metrics[cid]["recall"]
        
        print(f"\n[{name}]")
        print(f"  Dice:      {np.nanmean(dices):.4f} ± {np.nanstd(dices):.4f}")
        print(f"  HD95:      {np.nanmean(hd95s):.4f} ± {np.nanstd(hd95s):.4f} mm")
        print(f"  Precision: {np.nanmean(precs):.4f} ± {np.nanstd(precs):.4f}")
        print(f"  Recall:    {np.nanmean(recs):.4f} ± {np.nanstd(recs):.4f}")
    print("="*70)
    print(f"\n Successfully saved {total_slices} visualizations to: {out_dir}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir', type=str, default="/home/jovyan/seg_data_modified_2d/images")
    parser.add_argument('--mask_dir', type=str, default="/home/jovyan/seg_data_modified_2d/masks")
    parser.add_argument('--weights', type=str, default="seg_grid_search/Unet_CEDice_f64_lr0.0005_drop0.4_L20.01_annTrue_w0.1-0.9/best_model.pth")
    args = parser.parse_args()
    main(args)