
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops


__all__ = [
    "image_grid",
    "image_reshaper",
]


# TODO(joelshor): Make this a special case of `image_reshaper`.
def image_grid(input_tensor, grid_shape, image_shape=(32, 32), num_channels=3):
  
  if grid_shape[0] * grid_shape[1] != int(input_tensor.shape[0]):
    raise ValueError("Grid shape %s incompatible with minibatch size %i." %
                     (grid_shape, int(input_tensor.shape[0])))
  if len(input_tensor.shape) == 2:
    num_features = image_shape[0] * image_shape[1] * num_channels
    if int(input_tensor.shape[1]) != num_features:
      raise ValueError("Image shape and number of channels incompatible with "
                       "input tensor.")
  elif len(input_tensor.shape) == 4:
    if (int(input_tensor.shape[1]) != image_shape[0] or
        int(input_tensor.shape[2]) != image_shape[1] or
        int(input_tensor.shape[3]) != num_channels):
      raise ValueError("Image shape and number of channels incompatible with "
                       "input tensor.")
  else:
    raise ValueError("Unrecognized input tensor format.")
  height, width = grid_shape[0] * image_shape[0], grid_shape[1] * image_shape[1]
  input_tensor = array_ops.reshape(
      input_tensor, tuple(grid_shape) + tuple(image_shape) + (num_channels,))
  input_tensor = array_ops.transpose(input_tensor, [0, 1, 3, 2, 4])
  input_tensor = array_ops.reshape(
      input_tensor, [grid_shape[0], width, image_shape[0], num_channels])
  input_tensor = array_ops.transpose(input_tensor, [0, 2, 1, 3])
  input_tensor = array_ops.reshape(
      input_tensor, [1, height, width, num_channels])
  return input_tensor


def _validate_images(images):
  for img in images:
    img.shape.assert_has_rank(3)
    img.shape.assert_is_fully_defined()
    if img.shape[-1] not in (1, 3):
      raise ValueError("image_reshaper only supports 1 or 3 channel images.")


# TODO(joelshor): Move the dimension logic from Python to Tensorflow.
def image_reshaper(images, num_cols=None):
  
  if isinstance(images, ops.Tensor):
    images = array_ops.unstack(images)
  _validate_images(images)

  num_images = len(images)
  num_columns = (num_cols if num_cols else
                 int(math.ceil(math.sqrt(num_images))))
  num_rows = int(math.ceil(float(num_images) / num_columns))
  rows = [images[x:x+num_columns] for x in range(0, num_images, num_columns)]

  # Add empty image tiles if the last row is incomplete.
  num_short = num_rows * num_columns - num_images
  assert num_short >= 0 and num_short < num_columns
  if num_short > 0:
    rows[-1].extend([array_ops.zeros_like(images[-1])] * num_short)

  # Convert each row from a list of tensors to a single tensor.
  rows = [array_ops.concat(row, 1) for row in rows]

  # Stack rows vertically.
  img = array_ops.concat(rows, 0)

  return array_ops.expand_dims(img, 0)
