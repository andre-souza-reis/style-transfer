import tensorflow as tf

# Load model from local files
hub_module = tf.saved_model.load('./model')

def crop_center(image):
  # Returns a cropped square image.

  shape = image.shape
  new_shape = min(shape[1], shape[2])
  offset_y = max(shape[1] - shape[2], 0) // 2
  offset_x = max(shape[2] - shape[1], 0) // 2
  image = tf.image.crop_to_bounding_box(image, offset_y, offset_x, new_shape, new_shape)
  return image

def load_image(image, image_size=(256, 256)):
  # Loads and preprocesses images.

  # Adding batch dimension, cropping center and resizing.
  img = image[tf.newaxis, ...]
  img = crop_center(img)
  img = tf.image.resize(img, image_size, preserve_aspect_ratio=True)
  return img

def process_image(image, style):

  # Preferred content and image sizes
  content_image_size = 384
  style_image_size = 256

  # Preparing the input image
  content_img = load_image(image, (content_image_size, content_image_size))

  # Preparing the style image
  style_img = load_image(style, (style_image_size, style_image_size))

  # Stylize the content image using the style bottleneck.
  stylized_image = hub_module(tf.constant(
      content_img), tf.constant(style_img))[0]

  # Image processed
  return stylized_image.numpy()[0]*255