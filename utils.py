from matplotlib import gridspec
import matplotlib.pylab as plt
import tensorflow as tf
import numpy as np
import tensorflow_hub as hub
import random
import os

def crop_center(image):
  # Returns a cropped square image.

  shape = image.shape
  new_shape = min(shape[1], shape[2])
  offset_y = max(shape[1] - shape[2], 0) // 2
  offset_x = max(shape[2] - shape[1], 0) // 2
  image = tf.image.crop_to_bounding_box(image, offset_y, offset_x, new_shape, new_shape)
  return image

def load_local_image(image_url, image_size=(256, 256)):
  # Loads and preprocesses images.

  # Load and convert to float32 numpy array, add batch dimension, and normalize to range [0, 1].
  img = tf.io.decode_image(tf.io.read_file(image_url), channels=3, dtype=tf.float32)[tf.newaxis, ...]
  img = crop_center(img)
  img = tf.image.resize(img, image_size, preserve_aspect_ratio=True)
  return img

def load_image(image, image_size=(256, 256)):
  # Loads and preprocesses images.

  # Load and convert to float32 numpy array, add batch dimension, and normalize to range [0, 1].
  img = image[tf.newaxis, ...]
  img = crop_center(img)
  img = tf.image.resize(img, image_size, preserve_aspect_ratio=True)
  return img

def process_image(image):
  # Download the style bottleneck and transfer networks
  hub_module = hub.load(
    'https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')

  # Preferred content and image sizes
  content_image_size = 384
  style_image_size = 256

  # Selecting a random file as the first style picture
  style_counter = random.randrange(len(os.listdir("./Style_Reference_Images")))

  # Preparing the input image
  content_img = load_image(image, (content_image_size, content_image_size))

  # Choose the random image in the sequence
  style_name = os.listdir(
      "./Style_Reference_Images")[style_counter % len(os.listdir("./Style_Reference_Images"))]

  # Preparing the style image
  style_img = load_local_image(
      f"./Style_Reference_Images/{style_name}", (style_image_size, style_image_size))

  # Stylize the content image using the style bottleneck.
  stylized_image = hub_module(tf.constant(
      content_img), tf.constant(style_img))[0]

  # Image processed
  return stylized_image.numpy()[0]