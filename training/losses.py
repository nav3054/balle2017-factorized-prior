import tensorflow as tf

def compute_rate_distortion_loss(x, x_hat, bits, lambda_rd):
    """Calculates the rate-distortion loss."""
    
    # Total number of bits divided by total number of pixels.
    num_pixels = tf.cast(tf.reduce_prod(tf.shape(x)[:-1]), bits.dtype)
    bpp = tf.reduce_sum(bits) / num_pixels 

    # Distortion using Mean Squared Error (MSE)
    distortion = tf.reduce_mean(tf.math.squared_difference(x, x_hat))
    distortion = tf.cast(distortion, dtype=bpp.dtype)

    # Total loss: weighted sum of distortion and rate
    total_loss = lambda_rd * distortion + bpp

    return total_loss, (bpp, distortion)
