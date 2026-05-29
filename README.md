# 2D U-Net Medical Image Segmentation

A PyTorch implementation of a 2D U-Net neural network for medical image segmentation, specifically designed for MRI image analysis. This project focuses on segmenting anatomical structures from grayscale medical images using deep learning with comprehensive grid search and cross-validation support.
## Environment Setup

This project uses a Python virtual environment (`venv`).

### 1. Create environment

```bash
python3 -m venv p_env
source p_env/bin/activate
pip install -r requirements.txt
``` 


## Table of Contents

- [How It Works](#how-it-works)
- [Repository Structure](#repository-structure)
- [File Organization](#file-organization)
- [File-by-File Breakdown](#file-by-file-breakdown)
- [Output Directories](#output-directories)
- [Training Workflow](#training-workflow)

---

## How It Works

### The Big Picture

1. **Input**: Grayscale MRI images (224×224 pixels)
2. **Model**: U-Net neural network that learns to segment images into 3 classes:
   - **Class 0**: Background (no structure)
   - **Class 1**: STN (first anatomical structure)
   - **Class 2**: RN (second anatomical structure)
3. **Output**: Segmentation masks showing which pixels belong to which class
4. **Training**: Uses grid search to find the best hyperparameters (learning rate, dropout, loss function, etc.)

### The U-Net Architecture

The U-Net follows an **encoder-decoder** structure:

```
ENCODER (Compression - going down):
  Input Image (1×224×224)
      ↓
  Conv → ReLU → Conv → ReLU (32 channels)
      ↓ Pool (downsample)
  Conv → ReLU → Conv → ReLU (64 channels)
      ↓ Pool (downsample)
  Conv → ReLU → Conv → ReLU (128 channels)
      ↓ Pool (downsample)
  Conv → ReLU → Conv → ReLU (256 channels)
      ↓ Pool (downsample)

BOTTLENECK (Deepest point):
  Conv → ReLU → Conv → ReLU (512 channels)

DECODER (Reconstruction - going up):
  UpConv (512→256) + Skip Connection from Encoder
      ↓
  Conv → ReLU → Conv → ReLU (256 channels)
      ↓ UpConv (256→128) + Skip Connection from Encoder
      ↓
  Conv → ReLU → Conv → ReLU (128 channels)
      ↓ UpConv (128→64) + Skip Connection from Encoder
      ↓
  Conv → ReLU → Conv → ReLU (64 channels)
      ↓ UpConv (64→32) + Skip Connection from Encoder
      ↓
  Conv → ReLU → Conv → ReLU (32 channels)
      ↓
  Final Conv (32→3)
      ↓
  Output Segmentation (3×224×224)
```

---

## Repository Structure

```
2D-Unet/
│
├── Core Model Files
│   ├── unet.py                          # Neural network architecture
│   ├── losses.py                        # Loss functions for training
│   └── seg_dataset.py                   # Data loading & augmentation
│
├── Training & Optimization
│   ├── train_unet.py                    # Main training with grid search
│   └── crossValidation.py               # K-fold cross-validation
│
├── Evaluation & Analysis
│   ├── eval_unet_vis.py                 # Model evaluation & visualization
│   ├── valUnet.py                       # Validation metrics on 2D sclices and 3D volume
│   ├── valCrossUnet.py                  # Cross-validation evaluation
│   └── visualizeUnet.py                 # Prediction visualizations
│
├── Notebooks & Visualization
│   └── visualizeLoss.ipynb              # Jupyter notebook for curves
│
├── Output Directories (auto-created)
    ├── seg_cv_results/                  # Cross-validation results
    └── seg_grid_searchNewLosses/        # Grid search results

```

---

## File Organization

### Quick Reference Table

| File | Type | Purpose | Creates |
|------|------|---------|---------|
| `unet.py` | Model | Defines U-Net architecture | UNet class |
| `seg_dataset.py` | Data | Loads images, applies augmentation | MRISegDataset class |
| `losses.py` | Training | Custom loss functions | Loss classes |
| `train_unet.py` | Script | Main training + grid search | Model files + results |
| `eval_unet_vis.py` | Script | Evaluate + visualize | Metrics + images |
| `valUnet.py` | Module | Calculate validation metrics | Dice scores |
| `valCrossUnet.py` | Module | Aggregate cross-val metrics | Mean ± std stats |
| `crossValidation.py` | Script | K-fold cross-validation | CV results |
| `visualizeUnet.py` | Module | Create prediction overlays | Visual comparisons |
| `visualizeLoss.ipynb` | Notebook | Plot training curves | Graphs |

---

## File-by-File Breakdown

### 1. `unet.py` - The Neural Network Model

**Mission:** Defines the U-Net architecture that performs image segmentation.

**Contains:**
- `DoubleConv` class: Building block with 2 convolutions, batch norm, and ReLU
- `UNet` class: Full network with encoder, bottleneck, and decoder

**How it works:**
- Takes 1-channel grayscale image → compresses 4 times → expands 4 times with skip connections → outputs 3-channel segmentation
- Supports configurable feature channels, dropout rate, and input dimensions

**Key Parameters:**
```python
UNet(
    in_channels=1,          # Grayscale input
    num_classes=3,          # Background, STN, RN
    features=[32, 64, 128, 256],  # Channels per level
    dropout_rate=0.2        # Dropout for regularization
)
```

---

### 2. `seg_dataset.py` - Data Loading & Preprocessing

**Mission:** Loads images and masks, prevents data leakage, applies augmentation.

**Key Functions:**
- `extract_patient_id()`: Parses filenames to identify patient
  - Example: `P001_z00.png` → `P001`
- `get_patient_splits()`: **CRITICAL** - splits by PATIENT, not individual slices
  - Prevents data leakage (same patient doesn't appear in train AND test)
  - Only includes patients with annotated slices
  - Returns: `(train_patients, val_patients, test_patients)` sets

**Data Augmentation (training only):**
- Random horizontal/vertical flips
- Random rotation (±15°)
- Brightness & contrast adjustment
- Gaussian noise injection
- Gaussian filtering (smoothing)

**Output Format:**
```python
{
    'image': tensor(1, 224, 224),  # Grayscale image normalized 0-1
    'label': tensor(224, 224)       # Class indices: 0, 1, or 2
}
```

**Why Patient-Level Splitting?**
---

### 3. `losses.py` - Loss Functions

**Mission:** Defines different ways to measure prediction errors during training.

**Three Loss Functions:**

1. **CEDiceLoss**

2. **FocalTverskyLoss**

3. **UnifiedFocalLoss**

**All support class weights:**
```python
class_weights = torch.tensor([0.01, 0.5, 1.0])
# Background gets low weight, structures get high weight
```

---

### 4. `train_unet.py` - Main Training Script

**Mission:** Orchestrates training with grid search to find optimal hyperparameters.

**What It Does:**

1. **Early Stopping**: Stops training if validation performance doesn't improve
   - Saves best checkpoint automatically
   - Default patience: 15 epochs

2. **Grid Search**: Tests 72+ configurations combining:
   - Loss functions: CEDice, FocalTversky, UnifiedFocal (3 options)
   - Network depth: [32, 64, 128, 256] vs [64, 128, 256, 512] (2 options)
   - Learning rate: 1e-4, 5e-4 (2 options)
   - Weight decay: 1e-4, 1e-2 (2 options)
   - Dropout: 0.2, 0.4 (2 options)
   - Loss weights: (0.3, 0.7), (0.5, 0.5), (0.1, 0.9) (3 options)

3. **Training Loop** (for each configuration):
   - Trains on training set
   - Evaluates on validation set
   - Saves best checkpoint
   - Tests on test set

4. **Generates Reports:**
   - CSV file with all results (sortable by Dice score)
   - JSON files with detailed metrics per run

**Outputs Per Configuration:**
```
seg_grid_searchNewLosses/
└── Unet_CEDice_f32_lr0.0001_drop0.2_L20.0001_annTrue_w0.3-0.7/
    ├── best_model.pth        # Trained weights
    ├── train.log             # Per-epoch training log
    ├── history.json          # Training/validation curves
    └── results.json          # Final test metrics
```

**Usage:**
```bash
python train_unet.py \
  --image_dir /path/to/images \
  --mask_dir /path/to/masks \
  --output_dir ./seg_grid_searchNewLosses \
  --batch_size 32 \
  --epochs 200
```

---

### 5. `eval_unet_vis.py` - Evaluation & Visualization

**Mission:** Tests trained model and creates visual comparisons.

**Generates:**
- Segmentation predictions on test images
- Metric calculations (Dice scores, accuracy)
- Visual overlays of predictions vs. ground truth
- Grid images comparing multiple examples

**Output:** Visualization images showing:
```
Original Image | Ground Truth Mask | Model Prediction
```

---

### 6. `valUnet.py` - Validation Metrics

**Mission:** Calculates detailed segmentation performance metrics on both 2D slices and 3D volume.

**Computes:**
- **Dice Coefficient** per class (overlap between prediction and ground truth)
  - Formula: `2 * |A ∩ B| / (|A| + |B|)`
  - Range: 0 (no overlap) to 1 (perfect match)
- **Intersection over Union (IoU)** per class
- **Per-pixel accuracy**
- Handles multi-class evaluation separately for each class

**Metrics for 3 classes:**
- Class 0: Background Dice
- Class 1: STN Dice
- Class 2: RN Dice

---

### 7. `valCrossUnet.py` - Cross-Validation Evaluation

**Mission:** Aggregates metrics across multiple validation folds.

**Computes:**
- Mean metrics across all folds
- Standard deviation (shows variability)
- Confidence estimates of model performance

---

### 8. `crossValidation.py` - Cross-Validation Framework

**Mission:** Implements 5 ShuffleSplit for robust evaluation.
    ↓
Average results across all 5 folds

---

### 9. `visualizeUnet.py` - Prediction Visualization

**Mission:** Creates side-by-side visual comparisons of predictions.

**Generates:**
- Original image + ground truth mask + model prediction
- Color-coded visualizations
- Grid layouts showing multiple examples
- Helps identify failure cases visually

---

### 10. `visualizeLoss.ipynb` - Jupyter Notebook

**Mission:** Interactive visualization of training dynamics.

**Shows:**
- Training vs. validation loss curves over epochs
- Dice score evolution during training
- Comparison of different best configurations based on the finding (check the thesis for more information)
- Identifies overfitting, underfitting, or optimal stopping points


---

## Output Directories

### `seg_grid_searchNewLosses/` - Grid Search Results

Created during `train_unet.py` execution. Contains results from all configurations:

```
seg_grid_searchNewLosses/
├── Unet_CEDice_f32_lr0.0001_drop0.2_L20.0001_annTrue_w0.3-0.7/
│   ├── best_model.pth          # Trained model weights
│   ├── train.log               # Text log of all epochs
│   ├── history.json            # Per-epoch metrics (for plotting)
│   └── results.json            # Final test performance
│
├── Unet_FocalTversky_f64_lr0.0005_drop0.4_L20.01_annTrue_w0.5-0.5/
│   ├── best_model.pth
│   ├── train.log
│   ├── history.json
│   └── results.json
│
├── ... (more configurations)
│
├── grid_search_3d_summary.csv # ALL val/test in 2D slices/3D volume results 
├── grid_search_3d_summary.json # ALL val/test in 2D slices/3D volume results in JSOn format
├── grid_search_summary.csv     # ALL test 2D slices results in table (easy to sort)
└── grid_search_summary.json    # ALL 2D slices results in JSON format

```
**The full folder will be available in this google drive link: `link`**
---


### `seg_cv_results/` - Cross-Validation Results

Created during `crossValidation.py` execution:

```
seg_cv_results/
├── fold_0/
│   ├── best_model.pth
│   ├── results.json
│   └── history.json
│
├── fold_1/
│   ├── best_model.pth
│   ├── results.json
│   └── history.json
│
├── ... (fold_2, fold_3, fold_4)
│
└── cv_summary.json             # Mean ± std metrics across all folds
```
**The full folder will be available in this google drive link: `link`**
---

## Training Workflow

### Complete Step-by-Step Process

#### **Step 1: Prepare The preprocessed Data**
Organize your MRI images:
```
seg_data_modified_2d/
├── images/
│   ├── P001_z00.png     # Patient 001, slice 00
│   ├── P001_z01.png     # Patient 001, slice 01
│   ├── P001_z02.png     # Patient 001, slice 02
│   ├── P002_z00.png     # Patient 002, slice 00
│   └── ...
│
└── masks/
    ├── P001_z00.png     # Segmentation mask (0, 1, or 2)
    ├── P001_z01.png
    ├── P001_z02.png
    ├── P002_z00.png
    └── ...
```
**`Data will be provided on Request`**
**Filename Format:** `PatientID_zSliceNumber.png`
- Patient ID: `P001`, `P002`, `P012_1`, etc.
- Slice number: `z00`, `z01`, `z05`, etc.

#### **Step 2: Run Training with Grid Search**
```bash
python train_unet.py \
  --image_dir ./seg_data_modified_2d/images \
  --mask_dir ./seg_data_modified_2d/masks \
  --output_dir ./seg_grid_searchNewLosses \
  --batch_size 32 \
  --epochs 200
```

**What happens:**
- Tests all 144 configuration combinations
- For each configuration:
  - Creates training/val/test split (by patient)
  - Trains model (max 200 epochs)
  - Saves best model based on validation loss
- Creates summary files with all results

#### **Step 3: Analyze Results**
run ```bash python valUnet.py ```
Open `seg_grid_searchNewLosses/grid_search_3d_summary.csv`:
- Find the configuration with highest Dice score
- Note the directory name of best configuration

#### **Step 4: Examine Best Configuration**
```bash
# Open results
cat seg_grid_searchNewLosses/Unet_CEDice_f32_lr0.0001_drop0.2_L20.0001_annTrue_w0.3-0.7/results.json

# Plot training curves
# Open visualizeLoss.ipynb and load history.json from best run
jupyter notebook visualizeLoss.ipynb
```

#### **Step 5: Visualize Predictions**
```bash
python eval_unet_vis.py \
  --model_path ./seg_grid_searchNewLosses/best_config/best_model.pth \
  --image_dir ./seg_data_modified_2d/images \
  --mask_dir ./seg_data_modified_2d/masks
```

#### **Step 6: Cross-Validate (Optional - for robust metrics)**
```bash
python crossValidation.py \
  --image_dir ./seg_data_modified_2d/images \
  --mask_dir ./seg_data_modified_2d/masks \
  --folds 5
```

**Outputs:** `seg_cv_results/` with mean ± std metrics

---
## Summary

| Stage | File | Action | Output |
|-------|------|--------|--------|
| Data Prep | `seg_dataset.py` | Organize images/masks by patient | Directory structure |
| Training | `train_unet.py` | Run grid search (72 configs) | `seg_grid_searchNewLosses/` |
| Analysis | `grid_search_summary.csv` | Sort by Dice, find best | Best configuration name |
| Evaluation | `eval_unet_vis.py` | Test + visualize | Metrics + images |
| Validation | `crossValidation.py` | K-fold CV for robustness | Mean ± std metrics |
| Visualization | `visualizeLoss.ipynb` | Plot training curves | Graphs showing learning |

---

## Next Steps

1. **Organize your data** by patient ID (see Data Preparation section)
2. **Run training:** `python train_unet.py --image_dir ... --mask_dir ...`
3. **Check results:** Open `grid_search_summary.csv` for test only and `grid_search_3d_summary.csv` for validation and test
4. **Visualize best:** Use `eval_unet_vis.py` on winning configuration
5. **Cross-validate:** Run `crossValidation.py` for robust metrics
6. **Analyze curves:** Open `visualizeLoss.ipynb` with best `history.json`

---
