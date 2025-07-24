import numpy as np
import pandas as pd
import tensorflow as tf  
from tensorflow.python.ops.gen_array_ops import check_numerics
import tensorflow_compression as tfc
import argparse
import glob



def main():

    # create_dataset function call from data script


    # create model first


    # optimizer
    # compile

    # callbacks from callback script


    # model.fit()
    # args to be passed to .fit :- epochs, steps_per_epoch, validation_data, callbacks, verbose

    # model.save





if __name__ == "__main__":
    train(args)
    parser = argparse.ArgumentParser()

    # Data args
    data_dir
    patch_size
    batch_size
    patches_per_image
    max_images
    val_split
    #test_split # maybe remove this from here because we don't need this many 
               # images for testing -> it'll take too much time
               # let's take like 10 random images from training data for testing

    # Model args
    num_filters

    # Training args
    lambda_rd
    epochs
    steps_per_epoch
    precision_policy
    check_numerics





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


































