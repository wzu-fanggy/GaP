
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib.gan.python import namedtuples
from tensorflow.contrib.gan.python.eval.python import eval_utils
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import map_fn
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops.losses import util as loss_util
from tensorflow.python.summary import summary

__all__ = [
    'add_gan_model_image_summaries',
    'add_image_comparison_summaries',
    'add_gan_model_summaries',
    'add_regularization_loss_summaries',
    'add_cyclegan_image_summaries',
    'add_stargan_image_summaries'
]


def _assert_is_image(data):
  data.shape.assert_has_rank(4)
  data.shape[1:].assert_is_fully_defined()


def add_gan_model_image_summaries(gan_model, grid_size=4, model_summaries=True):
  
  if isinstance(gan_model, namedtuples.CycleGANModel):
    raise ValueError(
        '`add_gan_model_image_summaries` does not take CycleGANModels. Please '
        'use `add_cyclegan_image_summaries` instead.')
  _assert_is_image(gan_model.real_data)
  _assert_is_image(gan_model.generated_data)

  num_images = grid_size ** 2
  real_image_shape = gan_model.real_data.shape.as_list()[1:3]
  generated_image_shape = gan_model.generated_data.shape.as_list()[1:3]
  real_channels = gan_model.real_data.shape.as_list()[3]
  generated_channels = gan_model.generated_data.shape.as_list()[3]

  summary.image(
      'real_data',
      eval_utils.image_grid(
          gan_model.real_data[:num_images],
          grid_shape=(grid_size, grid_size),
          image_shape=real_image_shape,
          num_channels=real_channels),
      max_outputs=1)
  summary.image(
      'generated_data',
      eval_utils.image_grid(
          gan_model.generated_data[:num_images],
          grid_shape=(grid_size, grid_size),
          image_shape=generated_image_shape,
          num_channels=generated_channels),
      max_outputs=1)

  if model_summaries:
    add_gan_model_summaries(gan_model)


def add_cyclegan_image_summaries(cyclegan_model):
  
  if not isinstance(cyclegan_model, namedtuples.CycleGANModel):
    raise ValueError('`cyclegan_model` was not a CycleGANModel. Instead, was '
                     '%s' % type(cyclegan_model))

  _assert_is_image(cyclegan_model.model_x2y.generator_inputs)
  _assert_is_image(cyclegan_model.model_x2y.generated_data)
  _assert_is_image(cyclegan_model.reconstructed_x)
  _assert_is_image(cyclegan_model.model_y2x.generator_inputs)
  _assert_is_image(cyclegan_model.model_y2x.generated_data)
  _assert_is_image(cyclegan_model.reconstructed_y)

  def _add_comparison_summary(gan_model, reconstructions):
    image_list = (array_ops.unstack(gan_model.generator_inputs[:1]) +
                  array_ops.unstack(gan_model.generated_data[:1]) +
                  array_ops.unstack(reconstructions[:1]))
    summary.image(
        'image_comparison', eval_utils.image_reshaper(
            image_list, num_cols=len(image_list)), max_outputs=1)

  with ops.name_scope('x2y_image_comparison_summaries'):
    _add_comparison_summary(
        cyclegan_model.model_x2y, cyclegan_model.reconstructed_x)
  with ops.name_scope('y2x_image_comparison_summaries'):
    _add_comparison_summary(
        cyclegan_model.model_y2x, cyclegan_model.reconstructed_y)


def add_image_comparison_summaries(gan_model, num_comparisons=2,
                                   display_diffs=False):
  age(gan_model.generator_inputs)
  _assert_is_image(gan_model.generated_data)
  _assert_is_image(gan_model.real_data)

  gan_model.generated_data.shape.assert_is_compatible_with(
      gan_model.generator_inputs.shape)
  gan_model.real_data.shape.assert_is_compatible_with(
      gan_model.generated_data.shape)

  image_list = []
  image_list.extend(
      array_ops.unstack(gan_model.generator_inputs[:num_comparisons]))
  image_list.extend(
      array_ops.unstack(gan_model.generated_data[:num_comparisons]))
  image_list.extend(array_ops.unstack(gan_model.real_data[:num_comparisons]))
  if display_diffs:
    generated_list = array_ops.unstack(
        gan_model.generated_data[:num_comparisons])
    real_list = array_ops.unstack(gan_model.real_data[:num_comparisons])
    diffs = [
        math_ops.abs(math_ops.cast(generated, dtypes.float32) -
                     math_ops.cast(real, dtypes.float32))
        for generated, real in zip(generated_list, real_list)
    ]
    image_list.extend(diffs)

  # Reshape image and display.
  summary.image(
      'image_comparison',
      eval_utils.image_reshaper(image_list, num_cols=num_comparisons),
      max_outputs=1)


def add_stargan_image_summaries(stargan_model,
                                num_images=2,
                                display_diffs=False):
  

  _assert_is_image(stargan_model.input_data)
  stargan_model.input_data_domain_label.shape.assert_has_rank(2)
  stargan_model.input_data_domain_label.shape[1:].assert_is_fully_defined()

  num_domains = stargan_model.input_data_domain_label.get_shape().as_list()[-1]

  def _build_image(image):
    """Helper function to create a result for each image on the fly."""

    # Expand the first dimension as batch_size = 1.
    images = array_ops.expand_dims(image, axis=0)

    # Tile the image num_domains times, so we can get all transformed together.
    images = array_ops.tile(images, [num_domains, 1, 1, 1])

    # Create the targets to 0, 1, 2, ..., num_domains-1.
    targets = array_ops.one_hot(list(range(num_domains)), num_domains)

    with variable_scope.variable_scope(
        stargan_model.generator_scope, reuse=True):

      # Add the original image.
      output_images_list = [image]

      # Generate the image and add to the list.
      gen_images = stargan_model.generator_fn(images, targets)
      gen_images_list = array_ops.split(gen_images, num_domains)
      gen_images_list = [
          array_ops.squeeze(img, axis=0) for img in gen_images_list
      ]
      output_images_list.extend(gen_images_list)

      # Display diffs.
      if display_diffs:
        diff_images = gen_images - images
        diff_images_list = array_ops.split(diff_images, num_domains)
        diff_images_list = [
            array_ops.squeeze(img, axis=0) for img in diff_images_list
        ]
        output_images_list.append(array_ops.zeros_like(image))
        output_images_list.extend(diff_images_list)

      # Create the final image.
      final_image = eval_utils.image_reshaper(
          output_images_list, num_cols=num_domains + 1)

    # Reduce the first rank.
    return array_ops.squeeze(final_image, axis=0)

  summary.image(
      'stargan_image_generation',
      map_fn.map_fn(
          _build_image,
          stargan_model.input_data[:num_images],
          parallel_iterations=num_images,
          back_prop=False,
          swap_memory=True),
      max_outputs=num_images)


def add_gan_model_summaries(gan_model):
  """Adds typical GANModel summaries.

  Args:
    gan_model: A GANModel tuple.
  """
  if isinstance(gan_model, namedtuples.CycleGANModel):
    with ops.name_scope('cyclegan_x2y_summaries'):
      add_gan_model_summaries(gan_model.model_x2y)
    with ops.name_scope('cyclegan_y2x_summaries'):
      add_gan_model_summaries(gan_model.model_y2x)
    return

  with ops.name_scope('generator_variables'):
    for var in gan_model.generator_variables:
      summary.histogram(var.name, var)
  with ops.name_scope('discriminator_variables'):
    for var in gan_model.discriminator_variables:
      summary.histogram(var.name, var)


def add_regularization_loss_summaries(gan_model):
  """Adds summaries for a regularization losses..

  Args:
    gan_model: A GANModel tuple.
  """
  if isinstance(gan_model, namedtuples.CycleGANModel):
    with ops.name_scope('cyclegan_x2y_regularization_loss_summaries'):
      add_regularization_loss_summaries(gan_model.model_x2y)
    with ops.name_scope('cyclegan_y2x_regularization_loss_summaries'):
      add_regularization_loss_summaries(gan_model.model_y2x)
    return

  if gan_model.generator_scope:
    summary.scalar(
        'generator_regularization_loss',
        loss_util.get_regularization_loss(gan_model.generator_scope.name))
  if gan_model.discriminator_scope:
    summary.scalar(
        'discriminator_regularization_loss',
        loss_util.get_regularization_loss(gan_model.discriminator_scope.name))
