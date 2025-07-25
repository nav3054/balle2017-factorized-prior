import os
import tarfile
import zipfile
import random
import tensorflow as tf
from PIL import Image
from glob import glob

AUTOTUNE = tf.data.AUTOTUNE

#  test zip and extraction paths
CLIC_TEST2024_ZIPPED = "/content/drive/MyDrive/IMAGE_COMPRESSION/data/clic2024_test_image.zip"
CLIC_TEST2024_EXTRACTED = "/tmp/clic2024_test_images"

# training data extraction directory
TRAIN_EXTRACTED_DIR = "/tmp/train_images"


def extract_tar(tar_path, extract_path):
    if not os.path.exists(extract_path):
        os.makedirs(extract_path, exist_ok=True)
        print(f"[INFO] Extracting training tar {tar_path} to {extract_path} ...")
        with tarfile.open(tar_path, 'r') as tar:
            tar.extractall(path=extract_path)
        print("[INFO] Extraction complete.")
    else:
        print(f"[INFO] Training data already extracted at {extract_path}")


def extract_zip(zip_path, extract_path):
    if not os.path.exists(zip_path):
        raise FileNotFoundError(f"[ERROR] Test ZIP file not found at {zip_path}")

    if not os.path.exists(extract_path):
        os.makedirs(extract_path, exist_ok=True)
        print(f"[INFO] Extracting test zip {zip_path} to {extract_path} ...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_path)
        print("[INFO] Extraction complete.")
    else:
        print(f"[INFO] Test data already extracted at {extract_path}")


def load_image_paths(directory, patch_size):
    valid_exts = ('.jpg', '.jpeg', '.png')
    image_paths = glob(os.path.join(directory, '**'), recursive=True)
    filtered_paths = []

    for path in image_paths:
        if path.lower().endswith(valid_exts):
            try:
                with Image.open(path) as img:
                    if img.mode != 'RGB':
                        continue
                    width, height = img.size
                    if width >= patch_size and height >= patch_size:
                        filtered_paths.append(path)
            except Exception:
                continue

    return filtered_paths


def preprocess_image(image_path, patch_size):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_image(image, channels=3, expand_animations=False)
    image = tf.image.convert_image_dtype(image, tf.float32)

    shape = tf.shape(image)[:2]
    offset_height = tf.random.uniform((), 0, shape[0] - patch_size + 1, dtype=tf.int32)
    offset_width = tf.random.uniform((), 0, shape[1] - patch_size + 1, dtype=tf.int32)
    image = tf.image.crop_to_bounding_box(image, offset_height, offset_width, patch_size, patch_size)

    return image


def create_train_val_datasets(image_paths, patch_size, batch_size, val_split, patches_per_image):
    random.shuffle(image_paths)
    num_val = int(len(image_paths) * val_split)
    val_paths = image_paths[:num_val]
    train_paths = image_paths[num_val:]

    print(f"[INFO] Training images: {len(train_paths)}, Validation images: {len(val_paths)}")

    def expand_dataset(paths):
        def generator():
            for path in paths:
                for _ in range(patches_per_image):
                    yield path
        return tf.data.Dataset.from_generator(generator, output_types=tf.string)

    train_ds = expand_dataset(train_paths)
    val_ds = expand_dataset(val_paths)

    train_ds = train_ds.map(lambda x: preprocess_image(x, patch_size), num_parallel_calls=AUTOTUNE)
    val_ds = val_ds.map(lambda x: preprocess_image(x, patch_size), num_parallel_calls=AUTOTUNE)

    # caching OPTIONAL - ###### disable if RAM issues occur ######
    train_ds = train_ds.cache()
    val_ds = val_ds.cache()

    train_ds = train_ds.shuffle(buffer_size=1000)
    train_ds = train_ds.batch(batch_size, drop_remainder=True).prefetch(AUTOTUNE)

    val_ds = val_ds.batch(batch_size, drop_remainder=False).prefetch(AUTOTUNE)

    return train_ds, val_ds


def create_test_dataset(test_dir):
    valid_exts = ('.jpg', '.jpeg', '.png')
    image_paths = [f for f in glob(os.path.join(test_dir, '*')) if f.lower().endswith(valid_exts)]

    if not image_paths:
        raise RuntimeError(f"[ERROR] No valid test images found in {test_dir}")

    def load_and_preprocess(path):
        image = tf.io.read_file(path)
        image = tf.image.decode_image(image, channels=3, expand_animations=False)
        image = tf.image.convert_image_dtype(image, tf.float32)
        return image

    ds = tf.data.Dataset.from_tensor_slices(image_paths)
    ds = ds.map(load_and_preprocess, num_parallel_calls=AUTOTUNE)
    ds = ds.batch(1, drop_remainder=False).prefetch(AUTOTUNE)

    print(f"[INFO] Test images: {len(image_paths)}")

    return ds


def create_datasets(train_data_dir, patch_size, batch_size, patches_per_image=1, max_images=None, val_split=0.1):
    
    # Extract training data
    extract_tar(train_data_dir, TRAIN_EXTRACTED_DIR)

    # Extract test data
    extract_zip(CLIC_TEST2024_ZIPPED, CLIC_TEST2024_EXTRACTED)

    # Load image paths and filter the data
    all_train_paths = load_image_paths(TRAIN_EXTRACTED_DIR, patch_size)
    if max_images:
        all_train_paths = all_train_paths[:max_images]

    # Create datasets
    train_ds, val_ds = create_train_val_datasets(
        all_train_paths, patch_size, batch_size, val_split, patches_per_image
    )
    test_ds = create_test_dataset(CLIC_TEST2024_EXTRACTED)

    return train_ds, val_ds, test_ds













