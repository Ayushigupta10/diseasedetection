"""Shared RGB resize and normalization for both training and inference."""
import tensorflow as tf
IMAGE_SIZE=(224,224)
def preprocess_rgb(image):
    image=tf.cast(image,tf.float32)
    image=tf.ensure_shape(image,[None,None,3])
    image=tf.image.resize(image,IMAGE_SIZE,method='bilinear',antialias=False)
    return image/255.0
