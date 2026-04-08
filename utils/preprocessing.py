from PIL import Image
import numpy as np
import torch
import tensorflow as tf

def preprocess_pt(image):
    x = preprocess_base(image)
    return torch.tensor(x, dtype=torch.float32).permute(2, 0, 1)  # HWC → CHW

def preprocess_tf(image):
    return preprocess_base_tf(image) # stays HWC
    # return x  

def preprocess_base_tf(image):
    image = tf.cast(image, tf.float32) / 255.0

    mean = tf.constant([0.485, 0.456, 0.406])
    std = tf.constant([0.229, 0.224, 0.225])

    image = (image - mean) / std
    return image

def preprocess_base(image):
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

    # To numpy + normalize
    image = np.array(image).astype(np.float32) / 255.0

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    image = (image - mean) / std

    return image  # HWC