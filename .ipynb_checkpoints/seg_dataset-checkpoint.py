"""
Dataset for 2D MRI segmentation.
Critical design: split by PATIENT, not by slice — otherwise slices from
the same patient end up in both train and val (data leakage).

Patient IDs are inferred from filenames: P001_z00.png → patient P001
"""

import os
import re
import random
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
from scipy.ndimage import gaussian_filter

def extract_patient_id(filename):
    """P001_z00.png → 'P001',  P012_1_z05.png → 'P012_1'"""
    name = os.path.splitext(filename)[0]
    parts = re.split(r'_z\d+', name)
    return parts[0]

def get_patient_splits(mask_dir, val_fraction=0.2, test_fraction=0.0, seed=42):
    """
    Returns (train_patients, val_patients, test_patients) as sets of patient IDs.
    Only patients with at least one annotated slice are included.
    """
    annotated_patients = set()
    for fn in os.listdir(mask_dir):
        if not fn.endswith('.png'):
            continue
        mask = np.array(Image.open(os.path.join(mask_dir, fn)))
        if mask.max() > 0:
            annotated_patients.add(extract_patient_id(fn))

    patients = sorted(annotated_patients)
    random.seed(seed)
    random.shuffle(patients)

    n = len(patients)
    n_test  = int(n * test_fraction)
    n_val   = int(n * val_fraction)
    n_train = n - n_val - n_test

    train_p = set(patients[:n_train])
    val_p   = set(patients[n_train:n_train + n_val])
    test_p  = set(patients[n_train + n_val:])

    print(f"[Split] {n} annotated patients → train={len(train_p)}, val={len(val_p)}, test={len(test_p)}")
    return train_p, val_p, test_p

class MRISegDataset(Dataset):
    def __init__(self, image_dir, mask_dir, patient_ids,
                 img_size=224, augment=False, only_annotated=True):
        self.image_dir = image_dir
        self.mask_dir  = mask_dir
        self.img_size  = img_size
        self.augment   = augment

        self.samples = []
        for fn in sorted(os.listdir(mask_dir)):
            if not fn.endswith('.png'):
                continue
            pid = extract_patient_id(fn)
            if pid not in patient_ids:
                continue
            img_path = os.path.join(image_dir, fn)
            msk_path = os.path.join(mask_dir,  fn)
            if not os.path.exists(img_path):
                continue
            if only_annotated:
                mask = np.array(Image.open(msk_path))
                if mask.max() == 0:
                    continue
            self.samples.append((img_path, msk_path))

        print(f"[Dataset] {len(self.samples)} slices | patients={len(patient_ids)} | only_annotated={only_annotated}")

    def __len__(self):
        return len(self.samples)

    def _augment(self, image, mask):
        if random.random() < 0.5:
            image = TF.hflip(image);  mask = TF.hflip(mask)
        if random.random() < 0.5:
            image = TF.vflip(image);  mask = TF.vflip(mask)
        if random.random() < 0.5:
            angle = random.uniform(-15, 15)
            image = TF.rotate(image, angle, interpolation=TF.InterpolationMode.BILINEAR)
            mask  = TF.rotate(mask,  angle, interpolation=TF.InterpolationMode.NEAREST)
            
        if random.random() < 0.5:
            image = TF.adjust_brightness(image, random.uniform(0.8, 1.2))
        if random.random() < 0.5:
            image = TF.adjust_contrast(image, random.uniform(0.8, 1.2))
            
        if random.random() < 0.3:
            img_np = np.array(image).astype(np.float32) / 255.
            img_np = np.clip(img_np + np.random.normal(0, 0.02, img_np.shape), 0, 1)
            image  = Image.fromarray((img_np * 255).astype(np.uint8))
    
        if random.random() < 0.2:
            img_np = gaussian_filter(np.array(image).astype(np.float32), sigma=random.uniform(0.5, 1.0))
            image  = Image.fromarray(img_np.astype(np.uint8))
        return image, mask

    def __getitem__(self, idx):
        img_path, msk_path = self.samples[idx]

        image = Image.open(img_path).convert('L')
        mask  = Image.open(msk_path)

        image = TF.resize(image, [self.img_size, self.img_size], interpolation=TF.InterpolationMode.BILINEAR)
        mask  = TF.resize(mask,  [self.img_size, self.img_size], interpolation=TF.InterpolationMode.NEAREST)

        if self.augment:
            image, mask = self._augment(image, mask)

        image = TF.to_tensor(image)  # [1, H, W] float32
        mask  = torch.from_numpy(np.clip(np.round(np.array(mask)), 0, 2).astype(np.int64)) # [H, W] long
        return {'image': image, 'label': mask}