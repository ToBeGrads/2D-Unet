import argparse
import os
import random
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import distance_transform_edt
from tqdm import tqdm

from unet import UNet
from seg_dataset import get_patient_splits, MRISegDataset

def compute_hd95(pred, gt, spacing=(0.5, 0.5)):
    """Calculates the 95th percentile of the Hausdorff Distance."""
    if np.sum(pred) == 0 or np.sum(gt) == 0:
        return float('nan')

    # Distance transform of the ground truth and prediction
    dt_gt = distance_transform_edt(~gt.astype(bool), sampling=spacing)
    dt_pred = distance_transform_edt(~pred.astype(bool), sampling=spacing)
    
    # Distance from prediction to GT
    dist_p_to_g = dt_gt[pred.astype(bool)]
    # Distance from GT to prediction
    dist_g_to_p = dt_pred[gt.astype(bool)]
    
    hd95 = np.percentile(np.concatenate([dist_p_to_g, dist_g_to_p]), 95)
    return hd95

def compute_all_metrics(pred, true, class_idx):
    """Calculate core segmentation metrics for a specific class."""
    p = (pred == class_idx)
    t = (true == class_idx)
    
    tp = (p & t).sum()
    fp = (p & ~t).sum()
    fn = (~p & t).sum()
    tn = (~p & ~t).sum()
    
    union = p.sum() + t.sum()
    
    dice = float(2. * tp / union) if union > 0 else float('nan')
    iou = float(tp / (tp + fp + fn)) if (tp + fp + fn) > 0 else float('nan')
    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else float('nan')
    recall = float(tp / (tp + fn)) if (tp + fn) > 0 else float('nan')
    acc = float((tp + tn) / (tp + tn + fp + fn))
    
    hd95 = compute_hd95(p, t)
        
    return {
        "dice": dice, "iou": iou, "precision": precision, 
        "recall": recall, "accuracy": acc, "hd95": hd95
    }

def mask_to_rgb(mask):
    h, w = mask.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    rgb[mask == 1] = [0, 150, 255] # STN -> Blue
    rgb[mask == 2] = [255, 50, 50] # RN  -> Red
    return rgb

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. Load the TEST split
    _, _, test_p = get_patient_splits(args.mask_dir, val_fraction=0.1, test_fraction=0.1)
    test_dataset = MRISegDataset(args.image_dir, args.mask_dir, test_p, augment=False, only_annotated=True)
    
    print(f"Total test slices to evaluate: {len(test_dataset)}")

    # 2. Initialize Model (Adjust features if your best model used [32, 64, 128, 256])
    model = UNet(in_channels=1, num_classes=3, features=[64, 128, 256, 512])
    model.load_state_dict(torch.load(args.weights, map_location=device))
    model.to(device).eval()

    # Containers for metrics
    results = {1: [], 2: []} # 1: STN, 2: RN
    
    # 3. Evaluation loop over ALL test slices
    print("\nRunning evaluation on full test set...")
    with torch.no_grad():
        for i in tqdm(range(len(test_dataset))):
            data = test_dataset[i]
            img = data['image'].unsqueeze(0).to(device)
            gt = data['label'].numpy()
            
            output = model(img)
            pred = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()
            
            # Compute metrics for each class
            for cid in [1, 2]:
                m = compute_all_metrics(pred, gt, cid)
                results[cid].append(m)

    # 4. Calculate and Print Means
    print("\n" + "="*40)
    print(" FINAL TEST SET MEAN METRICS")
    print("="*40)
    
    for cid, name in zip([1, 2], ["STN", "RN"]):
        dices = [r['dice'] for r in results[cid] if not np.isnan(r['dice'])]
        hd95s = [r['hd95'] for r in results[cid] if not np.isnan(r['hd95'])]
        precs = [r['precision'] for r in results[cid] if not np.isnan(r['precision'])]
        recs  = [r['recall'] for r in results[cid] if not np.isnan(r['recall'])]
        
        print(f"\n[{name}]")
        print(f"  Mean Dice:      {np.mean(dices):.4f}")
        print(f"  Mean HD95:      {np.mean(hd95s):.4f} mm")
        print(f"  Mean Precision: {np.mean(precs):.4f}")
        print(f"  Mean Recall:    {np.mean(recs):.4f}")

    # 5. Visualizing a few samples for qualitative check
    num_vis = len(test_dataset)
    indices = random.sample(range(len(test_dataset)), num_vis)
    
    fig, axes = plt.subplots(num_vis, 3, figsize=(15, 5 * num_vis))
    if num_vis == 1: axes = [axes]

    for row_idx, idx in enumerate(indices):
        data = test_dataset[idx]
        img_tensor = data['image'].unsqueeze(0).to(device)
        gt = data['label'].numpy()
        
        with torch.no_grad():
            pred = torch.argmax(model(img_tensor), dim=1).squeeze(0).cpu().numpy()
        
        m_stn = compute_all_metrics(pred, gt, 1)
        m_rn  = compute_all_metrics(pred, gt, 2)

        mri_img = data['image'].squeeze().numpy()
        axes[row_idx][0].imshow(mri_img, cmap='gray')
        axes[row_idx][0].set_title(f"Test Sample {idx}\nMRI Input")
        axes[row_idx][0].axis('off')

        axes[row_idx][1].imshow(mri_img, cmap='gray')
        axes[row_idx][1].imshow(mask_to_rgb(gt), alpha=0.4)
        axes[row_idx][1].set_title("Ground Truth\nBlue=STN, Red=RN")
        axes[row_idx][1].axis('off')

        axes[row_idx][2].imshow(mri_img, cmap='gray')
        axes[row_idx][2].imshow(mask_to_rgb(pred), alpha=0.4)
        axes[row_idx][2].set_title(f"Prediction\nSTN Dice: {m_stn['dice']:.2f} | RN Dice: {m_rn['dice']:.2f}")
        axes[row_idx][2].axis('off')

    plt.tight_layout()
    save_path = os.path.join(os.path.dirname(args.weights), "test_evaluation_vis.png")
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    print(f"\nVisualization of {num_vis} test samples saved to: {save_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir', type=str, default="/home/jovyan/seg_data_modified_2d/images")
    parser.add_argument('--mask_dir', type=str, default="/home/jovyan/seg_data_modified_2d/masks")
    parser.add_argument('--weights', type=str, default="seg_grid_search/Unet_CEDice_f64_lr0.0005_drop0.4_L20.01_annTrue_w0.1-0.9/best_model.pth")
    #parser.add_argument('--num_samples', type=int, default=len(test_dataset), help="Number of samples to visualize")
    args = parser.parse_args()
    main(args)
