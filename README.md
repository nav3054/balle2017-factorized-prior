<!-- # balle2017-factorized-prior -->

# End-to-end Optimized Image Compression (Ballé et al., 2017)

This repository contains an unofficial TensorFlow 2.x implementation of the paper [End-to-end Optimized Image Compression](https://arxiv.org/abs/1611.01704) by Ballé et al. (2017).

## Features

- Implements the factorized prior architecture for variational image compression.
- Supports training and evaluation on large-scale or custom datasets.
- Modular utilities for checkpointing, early stopping, and image logging.
- Configurable via command-line arguments for flexible experimentation.

## Repository Structure

```
balle2017-factorized-prior
  ├── README.md
  ├── LICENSE
  ├── data/
      └── imagenet_patch_loader.py
  ├── model/
      └── balle_2017_model.py
  └── training/
      ├── train.py
      ├── callbacks.py
      └── losses.py
```

---

## Usage

### Training

The main entry point is `training/train.py`.

#### Command-line Arguments

**Data arguments:**
- `--data_dir`: Path to the directory containing training images (**required**).
- `--patch_size`: Size of the square patches to extract from images for training (default: `256`).
- `--batch_size`: Number of image patches per batch (default: `16`).
- `--patches_per_image`: Number of patches to extract from each image (default: `1`).
- `--max_images`: Maximum number of images to load from the dataset (default: `None`; uses all images found).
- `--val_split`: Portion of data to use for validation (between 0 and 1, default: `0.1`).

**Model arguments:**
- `--num_filters`: Number of filters per layer (default: `128`).

**Training arguments:**
- `--lambda_rd`: Rate-distortion trade-off lambda (default: `0.01`).
- `--epochs`: Number of training epochs (default: `100`). *(One epoch here is defined by the number of steps given by `--steps_per_epoch`, not iterations over the full training dataset.)*
- `--steps_per_epoch`: Number of steps per epoch (i.e., number of batches, default: `1000`).
- `--learning_rate`: Learning rate (default: `1e-4`).
- `--precision_policy`: Mixed precision (`tf.keras.mixed_precision`) policy to use during training. Choices: `"float32"`, `"mixed_float16"`, `"mixed_bfloat16"`. (default: `None`)
- `--check_numerics`: Enable to crash on NaN or Inf values in the graph (flag).
- `--save_dir`: Directory to save model checkpoints and logs (default: `"./checkpoints"`).

**Callback arguments:**
- `--early_stopping_patience`: Number of epochs with no improvement after which training will be stopped (default: `10`).
- `--log_images_every`: Epoch interval for logging reconstructed images (e.g., `5` = log original vs reconstructed comparison after every 5 epochs; default: `5`).
- `--num_images_to_log`: Number of images to display and save in the ImageLogger callback for original vs reconstructed image comparison (default: `4`).


Example usage:
```bash
python training/train.py \
  --data_dir /path/to/images \
  --patch_size 256 \
  --batch_size 16 \
  --num_filters 128 \
  --lambda_rd 0.01 \
  --epochs 200 \
  --num_images_to_log 4
```

---

## Core Components

### 1. Data Loading: `data/imagenet_patch_loader.py`

- **Supported Formats**: `.jpg`, `.jpeg`, `.png`
- **Features**:
  - Extracts datasets from `.tar` or `.zip` archives.
  - Splits dataset into train and validation sets.
  - Builds TensorFlow datasets of image patches (random crops).

**Main Functions:**
- `extract_tar`, `extract_zip`: Extract images from archives.
- `load_image_paths`: Lists valid image files in a directory.
- `preprocess_image`: Loads and randomly crops patches from images.
- `create_train_val_datasets`: Returns train and validation datasets.

### 2. Model: `model/balle_2017_model.py`

- Implements the Ballé et al. (2017) Factorized Prior Model for learned image compression.
- **Key Classes**:
  - `BalleEncoder`: Encoder network using SignalConv2D and GDN activations.
  - `BalleDecoder`: Decoder network using SignalConv2D and inverse GDN.
  - `Balle2017FactorizedPrior`: Main model class with encoder, decoder, and factorized prior entropy model.
- **Loss Function**: Weighted rate-distortion loss, combining bits-per-pixel and distortion (MSE).

### 3. Training Script: `training/train.py`

- Parses arguments and loads data.
- Initializes and compiles the model.
- Sets up training callbacks (checkpointing, early stopping, image logging).
- Trains the model and saves final weights.
- Evaluates on test set and prints loss, bits-per-pixel, distortion.

### 4. Callbacks: `training/callbacks.py`

- **KeepLastNCheckpoints**: Retains only last N checkpoints during training.
- **ImageLoggingCallback**: Logs original vs reconstructed images at intervals.
- **EarlyStopping**: Stops training when validation loss does not improve.

### 5. Loss Function: `training/losses.py`

- Implements rate-distortion loss.
- **compute_rate_distortion_loss**: Combines bits-per-pixel (bpp OR rate) and distortion (MSE).

---

## Example Workflow

1. **Prepare Dataset**: Place images in a directory or compress into a tar/zip file.
2. **Configure Training**: Set arguments for patch size, batch size, model architecture, etc.
3. **Train Model**: Run `train.py` with desired arguments.
4. **Monitor Training**: Checkpoints, early stopping, and image logs available.
5. **Evaluate Model**: Final evaluation performed on test set.

---

## Outputs

- **Checkpoints**: Saved every N epochs; only last 5 are retained.
- **Final Model**: Saved in the specified save directory.
- **Image Logs**: Side-by-side original and reconstructed images for inspection.
- **Test Results**: Prints loss, bits-per-pixel (BPP), and distortion metrics.

---

## References

- Ballé, J., Laparra, V., & Simoncelli, E. P. (2017). "End-to-end Optimized Image Compression." [Paper link](https://arxiv.org/abs/1611.01704)

---

## Contributing

Feel free to fork the repo and submit pull requests.
