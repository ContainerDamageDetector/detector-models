import io
import os
import scipy.misc
import numpy as np
import six
import time
import glob
from IPython.display import display
import json
import boto3
from botocore.exceptions import ClientError

from six import BytesIO

import matplotlib
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont

import tensorflow as tf
from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

os.environ['AWS_ACCESS_KEY_ID'] = 'AKIA5OSYVVF3F5WPQHEY';
os.environ['AWS_SECRET_ACCESS_KEY'] = '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';
os.environ['AWS_REGION'] = 'ap-south-1';
os.environ['S3_USE_HTTPS'] = '1';
os.environ['S3_VERIFY_SSL'] = '1';

s3 = boto3.client('s3')

# loads an image from an Amazon S3 bucket and converts it into a numpy array
def load_image_into_numpy_array(s3_bucket, s3_key):
    response = s3.get_object(Bucket=s3_bucket, Key=s3_key)
    img = Image.open(BytesIO(response['Body'].read()))
    (im_width, im_height) = img.size
    return np.array(img.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)


#  runs the inference for a single image using a given TensorFlow model
def run_inference_for_single_image(model, image):
  image = np.asarray(image)
  # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
  input_tensor = tf.convert_to_tensor(image)
  # The model expects a batch of images, so add an axis with `tf.newaxis`.
  input_tensor = input_tensor[tf.newaxis,...]

  # Run inference
  model_fn = model.signatures['serving_default']
  output_dict = model_fn(input_tensor)

  # All outputs are batches tensors.
  # Convert to numpy arrays, and take index [0] to remove the batch dimension.
  # We're only interested in the first num_detections.
  num_detections = int(output_dict.pop('num_detections'))
  output_dict = {key:value[0, :num_detections].numpy()
                 for key,value in output_dict.items()}
  output_dict['num_detections'] = num_detections

  # detection_classes should be ints.
  output_dict['detection_classes'] = \
                            output_dict['detection_classes'].astype(np.int64)

  # Handle models with masks:
  if 'detection_masks' in output_dict:
    # Reframe the the bbox mask to the image size.
    detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
              output_dict['detection_masks'], output_dict['detection_boxes'],
               image.shape[0], image.shape[1])
    detection_masks_reframed = tf.cast(detection_masks_reframed > 0.5,
                                       tf.uint8)
    output_dict['detection_masks_reframed'] = detection_masks_reframed.numpy()

  return output_dict