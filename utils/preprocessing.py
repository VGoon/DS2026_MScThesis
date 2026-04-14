import torch
import numpy as np
from PIL import Image
import tensorflow as tf

# “Although preprocessing differs across frameworks, this reflects the original training conditions of the models and is necessary to ensure valid inference behavior.”

# --- PYTORCH ---
# wants rgb and normalized with mean/std (0-1 scale)
def preprocess_pytorch(image):
    image = preprocess_py_resize_crop(image)
    x = preprocess_py_normalize(image)
    return torch.tensor(x, dtype=torch.float32).permute(2, 0, 1)

def preprocess_py_mobilenet(image):
    image = preprocess_py_resize_crop(image)
    image = np.array(image).astype(np.float32)
    image = (image / 127.5) - 1.0
    return torch.tensor(image, dtype=torch.float32).permute(2, 0, 1)

# scale [-1-1] for both pytorch and tf
def preprocess_tf_mobilenet(image):
    image = preprocess_tf_resize_crop(image)
    image = tf.cast(image, tf.float32)
    return (image / 127.5) - 1.0

# --- TENSORFLOW ---
# vgg and resnet
# bgr, no std scaling, subtract pixel means (0-255 scale)
def preprocess_tf(image):
    image = preprocess_tf_resize_crop(image)
    image = tf.cast(image, tf.float32)

    # RGB → BGR
    image = image[..., ::-1]

    # subtract ImageNet mean
    mean = tf.constant([103.939, 116.779, 123.68])
    image = image - mean

    return image

def preprocess_py_resize_crop(image):
    # Resize (shorter side = 256)
    w, h = image.size
    if w < h:
        new_w = 256
        new_h = int(h * 256 / w)
    else:
        new_h = 256
        new_w = int(w * 256 / h)

    image = image.resize((new_w, new_h), Image.BILINEAR)

    # Center crop 224x224
    left = (new_w - 224) // 2
    top = (new_h - 224) // 2
    image = image.crop((left, top, left + 224, top + 224))

    return image

def preprocess_py_normalize(image):
    # To numpy + normalize
    image = np.array(image).astype(np.float32) / 255.0

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    image = (image - mean) / std
    return image

def preprocess_tf_resize_crop(image):
    image = tf.cast(image, tf.float32)

    # Resize so shorter side = 256
    shape = tf.shape(image)
    h, w = shape[1], shape[2]

    scale = 256.0 / tf.minimum(tf.cast(h, tf.float32), tf.cast(w, tf.float32))
    new_h = tf.cast(tf.round(tf.cast(h, tf.float32) * scale), tf.int32)
    new_w = tf.cast(tf.round(tf.cast(w, tf.float32) * scale), tf.int32)

    image = tf.image.resize(image, (new_h, new_w))

    # Center crop 224x224
    image = tf.image.resize_with_crop_or_pad(image, 224, 224)

    return image
