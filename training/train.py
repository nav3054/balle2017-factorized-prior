import numpy as np
import pandas as pd
import tensorflow as tf  
from tensorflow.python.ops.gen_array_ops import check_numerics
import tensorflow_compression as tfc
import argparse
import glob

from model.balle_2017_model import Balle2017FactorizedPrior
from training.callbacks import get_callbacks
from data.imagenet_patch_loader import create_datasets



def main():
    if args.precision_policy:
        tf.keras.mixed_precision.set_global_policy(args.precision_policy) # setting mixed precision plocy
    
    if args.check_numerics:
        tf.debugging.enable_check_numerics() # this stops training when NaN or Inf is encountered


    # create_dataset function call from data script
    train_ds, val_ds, test_ds = create_datasets(
        train_data_dir = args.data_dir, 
        patch_size = args.patch_size, 
        batch_size = args.batch_size, 
        patches_per_image = args.patches_per_image, 
        max_images = args.max_images, 
        val_split = args.val_split)

    # create model first
    model = Balle2017FactorizedPrior()

    # optimizer
    # compile

    # callbacks from callback script


    # model.fit()
    # args to be passed to .fit :- epochs, steps_per_epoch, validation_data, callbacks, verbose

    # model.save
    # save model here





if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # Data args
    parser.add_argument("--data_dir", type=str, required=True, help="Path to the directory containing training images")
    parser.add_argument("--patch_size", type=int, default=256, help="Size of the square patches to extract from images for training")
    parser.add_argument("--batch_size", type=int, default=16, help="Number of image patches per batch")
    parser.add_argument("--patches_per_image", type=int, default=1, help="Number of patches to extract from each image")
    parser.add_argument("--max_images", type=int, default=None, help="Maximum number of images to load from the dataset")
    parser.add_argument("--val_split", type=float, default=0.1, help="Portion of data to use for validation (between 0 and 1)")

    # Model args
    parser.add_argument("--num_filters", type=int, default=128, help="Number of filters per layer")

    # Training args
    parser.add_argument("--lambda_rd", type=float, default = 0.01, help="Rate-distortion trade-off lambda")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs (One epoch is here defined as the number of "
                        "steps given by --steps_per_epoch, not iterations over the full training dataset.)")
    parser.add_argument("--steps_per_epoch", type=int, default=1000, help="Number of steps per epoch (i.e. number of batches)")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="")
    parser.add_argument("--precision_policy", type=str, default=None, choices=["float32", "mixed_float16", "mixed_bfloat16"], 
                        help="Mixed precision (tf.keras.mixed_precision) policy to use during training")
    parser.add_argument("--check_numerics", action="store_true", help="Enable to crash on NaN or Inf values in the graph")
    parser.add_argument("--save_dir", type=str, default="./checkpoints", help="Directory to save model checkpoints and logs")

    # Callback args
    parser.add_argument("--early_stopping_patience", type=int, default=10, help="Number of epochs with no improvement after which training will be stopped")
    parser.add_argument("--log_images_every", type=int, default=5, help="Epoch interval for logging reconstructed images "
                        "(e.g., 5 = log original vs reconstructed comparison after every 5 epochs)")
    parser.add_argument("--num_images_to_log", type=int, default=4, help="Number of images to display and save in the ImageLogger callback/Number "
                        "of images for the Original vs Reconstrcuted image comparison")

    args = parser.parse_args()
    main(args)



'''

ADD IN THE DATA SCRIPT IN PROBABLY THE MAIN FUNCTION THAT'S CALLED BY train.py
    dataset = dataset.map(
        lambda x: crop_image(read_png(x), args.patchsize),
        num_parallel_calls=tf.data.AUTOTUNE) # Changed to AUTOTUNE

    # Batch the dataset. drop_remainder=True ensures all batches have
    # the same size, which can be beneficial for fixed-size inputs in models.
    dataset = dataset.batch(args.batchsize, drop_remainder=True)

    # Prefetch data to overlap data preprocessing and model execution.
    # This also uses AUTOTUNE to optimize the prefetching buffer size.
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

    return dataset

'''


































