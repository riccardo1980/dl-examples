import logging
import numpy

from neon.data import MNIST
import matplotlib.pyplot as plt

from LecunMNIST import LecunMNIST

from neon.util.argparser import NeonArgparser
from neon import logger as neon_logger

from neon.data.dataloaderadapter import DataLoaderAdapter
from neon.data.dataloader_transformers import TypeCast, OneHot, ValueNormalize
from aeon import DataLoader
from GreyValueNormalize import GreyValueNormalize

import numpy as np
import sys

from itertools import izip
"""Compare LecunMNIST and raw data loading to original neon MNIST

Example:
  python mnist_compare.py

  After decompressing MNIST (see mnist_to_png.py):
  python mnist_compare.py --source raw

"""

def shape_equal(a, b):
  return a.shape == b.shape

def content_equal(a, b):
  aa = a.get()
  bb = b.get()
  return np.linalg.norm(aa - bb) == 0

class ImagesContentException(Exception):
  pass

class ImagesShapeException(Exception):
  pass

class LabelsContentException(Exception):
  pass

class LabelsShapeException(Exception):
  pass

def from_raw():
  image_config =        {"type": "image",
                         "height": 28,
                         "width": 28,
                         "channels": 1}

  label_config =        {"type": "label",
                         "binary": True}

  augmentation_config = {"type": "image"}

  aeon_config_train = {"manifest_filename": "train/train.tsv",
                       "manifest_root": "./",
                       "etl": (image_config, label_config),
                       "batch_size": args.batch_size}

  aeon_config_valid = {"manifest_filename": "test/test.tsv",
                       "manifest_root": "./",
                       "etl": (image_config, label_config),
                       "batch_size": args.batch_size}

  train_set = DataLoaderAdapter(DataLoader(aeon_config_train))
  train_set = TypeCast(train_set, index = 0, dtype = np.float32)
  train_set = GreyValueNormalize(train_set, index = 0,
                             source_range = [0., 255.],
                             target_range = [0., 1.])
  train_set = OneHot(train_set, index = 1, nclasses = 10)

  valid_set = DataLoaderAdapter(DataLoader(aeon_config_valid))
  valid_set = TypeCast(valid_set, index = 0, dtype = np.float32)
  valid_set = GreyValueNormalize(valid_set, index = 0,
                             source_range = [0., 255.],
                             target_range = [0., 1.])
  valid_set = OneHot(valid_set, index = 1, nclasses = 10)

  return train_set, valid_set

def tuple_print(obj, tab=0):
  if isinstance(obj[0], tuple):
    for ii in range( len(obj) ):
      tuple_print(obj[ii], tab=tab+1)
  else:
    k,v = obj[0], obj[1]
    if isinstance(v, tuple):
      print(''.rjust(tab)+str(k))
      for ii in range(len(v)):
        tuple_print(v[ii], tab=tab+1)
    else:
      print(''.rjust(tab)+str(k)+': '+str(v))

def dict_print(obj, tab=1):
  for k,v in obj.iteritems():
    if isinstance(v, tuple):
      print(''.rjust(tab)+str(k))
      for ii in range(len(v)):
        dict_print(v[ii], tab=tab+1)
    else:
      print(''.rjust(tab)+str(k)+': '+str(v))

def print_aeon( dl ):
  print('batch_size: %d' % (dl.batch_size,))
  print('ndata: %d' % (dl.ndata,))
  print('axes_info:')
  tuple_print(dl.axes_info)
  print('config:')
  dict_print(dl.config)

def shape_resolve(batch, raw_batch):
  print('reference shape (images): %s' % (str(batch[0].shape),))
  print('raw shape (images)      : %s' % (str(raw_batch[0].shape),))
  print('reference shape (labels): %s' % (str(batch[1].shape),))
  print('raw shape (labels)      : %s' % (str(raw_batch[1].shape),))

def show_error(label, raw_label, image, raw_image, figure):
  figure.clear()
  ax1 = figure.add_subplot(121)
  ax1.imshow(image, cmap=plt.cm.Greys)
  ax1.title.set_text(str(label))
  ax2 = figure.add_subplot(122)
  ax2.imshow(raw_image, cmap=plt.cm.Greys)
  ax2.title.set_text(str(raw_label))

  return (ax1,ax2)

def content_resolve(batch, raw_batch):

  labels = np.argmax(batch[1].get(), axis=0)
  raw_labels = np.argmax(raw_batch[1].get(), axis=0)

  images = batch[0].get().transpose()
  raw_images = raw_batch[0].get().transpose()

  # scan and find first difference
  for idx in range(len(images)):
    label_flag = ( labels[idx] != raw_labels[idx] )
    err = np.linalg.norm(images[idx] - raw_images[idx])
    image_flag = ( err != 0.0 )

    if label_flag or image_flag:
      if idx > 0:
        # last correct one
        fig1 = plt.figure('Last correct')
        show_error(labels[idx-1], raw_labels[idx-1],
          images[idx-1].reshape(28,28), raw_images[idx-1].reshape(28,28),
          fig1)

      # error one
      fig2 = plt.figure('Error')
      show_error(labels[idx], raw_labels[idx],
          images[idx].reshape(28,28), raw_images[idx].reshape(28,28), fig2)
      print('error on example:     %d ' % (idx,))
      print('image L2 relative error: %e' % (err/np.linalg.norm(images[idx]),))

      # next one
      if idx < len(images) -1:
        # last correct one
        fig3 = plt.figure('Next one')
        show_error(labels[idx+1], raw_labels[idx+1],
          images[idx+1].reshape(28,28), raw_images[idx+1].reshape(28,28),
          fig3)
        print('error on example:     %d ' % (idx+1,))
        print('image L2 relative error: %e' % (np.linalg.norm(images[idx+1] -
          raw_images[idx+1])/np.linalg.norm(images[idx+1]),))

      plt.show()
      break

if __name__=='__main__':
  logger = logging.getLogger(__name__)

  # parse the command line arguments
  parser = NeonArgparser(__doc__)
  parser.add_argument('--source', type=str, default='lecun',
      help='dataset source [lecun|raw]')

  args = parser.parse_args()

  # reference
  dataset = MNIST(path=args.data_dir)
  train_set = dataset.train_iter
  valid_set = dataset.valid_iter

  # dataset retrieval system to be checked
  if args.source == 'lecun':
    rawdataset = MNIST(path=args.data_dir)
    raw_train_set = dataset.train_iter
    raw_valid_set = dataset.valid_iter
  else:
    if args.source == 'raw':
      raw_train_set, raw_valid_set = from_raw()
    else:
      print('source: %s not recognized' % (args.source,))
      sys.exit(-1)

  if args.source == 'raw':
    print_aeon(raw_train_set)

  for idx, (batch, raw_batch) in enumerate( izip( train_set, raw_train_set ) ):

    try:
      if shape_equal(batch[0], raw_batch[0]) == False:
        raise ImagesShapeException()

      if shape_equal(batch[1], raw_batch[1]) == False:
        raise LabelsShapeException()

      if content_equal(batch[0], raw_batch[0]) == False:
        raise ImagesContentException()

      if content_equal(batch[1], raw_batch[1]) == False:
        raise LabelsContentException()

    except (ImagesShapeException, LabelsShapeException):
      print('Shape error on minibatch %d of %d' % (idx, raw_train_set.nbatches))
      shape_resolve(batch, raw_batch)
      sys.exit(-1)

    except (ImagesContentException, LabelsContentException):
      print('Content error on minibatch %d of %d' % (idx, raw_train_set.nbatches))
      content_resolve(batch, raw_batch)
      sys.exit(-1)
