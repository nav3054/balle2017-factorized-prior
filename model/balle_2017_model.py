# Balle2017FactorizedPrior script

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class BalleEncoder(tf.keras.Sequential):
    "Encoder class"

    def __init__(self, num_filters):
        super().__init__(name = "encoder")
        
        self.add(tf.keras.layers.Lambda(lambda x: x / 255.))

        self.add(tfc.SignalConv2D(   # SignalConv2D is like Conv2D but specifically designed for compression purposes.
            num_filters, (9,9),corr=True, strides_down=4, padding="same_zeros", use_bias=True, 
            activation = tfc.GDN())) # GDN -> Generalized Divisive Normalization activation function
        
        self.add(tfc.SignalConv2D(
            num_filters, (5,5), corr=True, strides_down=2, padding="same_zeros", use_bias=True, 
            activation = tfc.GDN()))
        
        self.add(tfc.SignalConv2D(
            num_filters, (5,5), corr=True, strides_down=2, padding="same_zeros", use_bias=True, 
            activation = tfc.GDN()))


class BalleDecoder(tf.keras.Sequential):
    "Decoder class"

    def __init__(self, num_filters):
        super().__init__(name = "decoder")
        
        self.add(tfc.SignalConv2D(
            num_filters, (5,5), corr=False, strides_up=2, padding="same_zeros", use_bias=True, 
            activation = tfc.GDN(inverse=True))) # GDN(inverse=True) -> Inverse Generalized Divisive Normalization activation function
        
        self.add(tfc.SignalConv2D(
            num_filters, (5,5), corr=False, strides_up=2, padding="same_zeros", use_bias=True, 
            activation = tfc.GDN(inverse=True)))
        
        self.add(tfc.SignalConv2D(
            3, (9,9), corr=False, strides_up=4, padding="same_zeros", use_bias=True, 
            activation = None))

        self.add(tf.keras.layers.Lambda(lambda x : x * 255.))



class Balle2017FactorizedPrior(keras.Model):
    """Main model class - "End-to-End Optimized Image Compression" by Ballé et al. (2017)."""
    def __init__(self, num_filters, lambda_rd):
        super().__init__()
        self.num_filters = num_filters
        self.lambda_rd = lambda_rd

        self.encoder = BalleEncoder(num_filters)
        self.decoder = BalleDecoder(num_filters)
        self.prior = tfc.NoisyDeepFactorized(batch_shape=(num_filters,))
        self.entropy_model = tfc.ContinuousBatchedEntropyModel(
            self.prior, 
            coding_rank=3,
            compression=False)

        self.loss_metric = tf.keras.metrics.Mean(name="loss")
        self.bpp_metric = tf.keras.metrics.Mean(name="bpp")
        self.distortion_metric = tf.keras.metrics.Mean(name="distortion")


    def call(self, x, training):
        y = self.encoder(x) # encodes input image x into latent rep y
        y_hat, bits = self.entropy_model(y, training=training) # quantizes y and estimates bits using the entropy model, generating y_hat and bits  
        x_hat = self.decoder(y_hat) # reconstructs image x_hat from quantized latent y_hat
        return x_hat, bits


    def train_step(self, x):
        with tf.GradientTape() as tape:
            x_hat, bits = self(x, training=True)
            loss, (bpp, distortion) = compute_rate_distortion_loss(
                x, x_hat, bits,
                lambda_rd = self.lambda_rd
            )
        
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        # Update metrics
        self.loss_metric.update_state(loss)
        self.bpp_metric.update_state(bpp)
        self.distortion_metric.update_state(distortion)

        return {
        "loss": self.loss_metric.result(),
        "bpp": self.bpp_metric.result(),
        "distortion": self.distortion_metric.result()
        }

    def test_step(self, data):
        x = data  # assuming you pass only inputs during validation

        # Forward pass
        y = self.encoder(x)
        y_hat, bits = self.entropy_model(y, training=False)
        x_hat = self.decoder(y_hat)

        # Rate-distortion loss (with training=False)
        loss, logs = compute_rate_distortion_loss(
            x, x_hat, bits, self.lambda_rd
        )

        self.loss_metric.update_state(loss)
        self.bpp_metric.update_state(bpp)
        self.distortion_metric.update_state(distortion)

        return {
            "loss": self.total_loss_tracker.result(),
            "bpp": self.bpp_tracker.result(),
            "distortion": self.distortion_tracker.result(),
        }


    def reset_metrics(self):
        self.loss_metric.reset_state()
        self.bpp_metric.reset_state()
        self.distortion_metric.reset_state()

    @property
    def metrics(self):
        return [self.loss_metric, self.bpp_metric, self.distortion_metric]

    def compress(self, x):
        x = tf.expand_dims(x, 0)  # add batch dim
        x = tf.cast(x, self.compute_dtype)
        y = self.encoder(x)
        x_shape = tf.shape(x)[1:-1]
        y_shape = tf.shape(y)[1:-1]
        compressed = self.entropy_model.compress(y)
        return compressed, x_shape, y_shape

    @tf.function(input_signature=[
        tf.TensorSpec(shape=(None,), dtype=tf.string), # compressed bitstream is variable length string
        tf.TensorSpec(shape=(2,), dtype=tf.int32),  # original image spatial shape (H,W)
        tf.TensorSpec(shape=(2,), dtype=tf.int32)  # latent spatial shape (H,W)
    ])
    def decompress(self, compressed, x_shape, y_shape):
        y_hat = self.entropy_model.decompress(compressed, y_shape)
        x_hat = self.decoder(y_hat)
        x_hat = x_hat[0, :x_shape[0], :x_shape[1], :]  # crop if any padding
        return tf.saturate_cast(tf.round(x_hat), tf.uint8)

