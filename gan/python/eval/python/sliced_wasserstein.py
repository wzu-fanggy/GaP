
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib.gan.python.eval.python import sliced_wasserstein_impl
# pylint: disable=wildcard-import
from tensorflow.contrib.gan.python.eval.python.sliced_wasserstein_impl import *
# pylint: enable=wildcard-import
from tensorflow.python.util.all_util import remove_undocumented

__all__ = sliced_wasserstein_impl.__all__
remove_undocumented(__name__, __all__)
