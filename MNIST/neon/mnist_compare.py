import logging
import numpy

from neon.data import MNIST
import matplotlib.pyplot as plt

from LecunMNIST import LecunMNIST

from neon.util.argparser import NeonArgparser
from neon import logger as neon_logger

from neon.data.dataloaderadapter import DataLoaderAdapter
from neon.data.dataloader_transformers import TypeCast, OneHot
from aeon import DataLoader
from GreyValueNormalize import GreyValueNormalize

import numpy as np
import sys

from itertools import izip
"""Compare LecunMNIST and raw data loading to original neon MNIST

Example:
  python mnist_compare.py

  python mnist_compare.py -g -p 0.01

  After decompressing MNIST (see mnist_to_png.py):
  python mnist_compare.py --source raw

  python mnist_compare.py --source raw -g -p 0.01

"""
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

if __name__=='__main__':
  logger = logging.getLogger(__name__)

  # parse the command line arguments
  parser = NeonArgparser(__doc__)
  parser.add_argument('-g', '--graphic', action='store_true',
      help='visualize images/labels')
  parser.add_argument('-p', '--pause', type=float, default=0.2,
      help='pause between image visualizations')
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

  if args.graphic == True:
    _, (ax1, ax2) = plt.subplots(1, 2)

  for idx, ((x, t), (rx, rt)) in enumerate( izip( train_set, raw_train_set ) ):

      np_x = x.get()
      np_rx = rx.get()
      if np_x.shape != np_rx.shape:
        print('data shape error (minibatch %d)!' % (idx,))
        print('data shape: ' + str( np_x.shape ) )
        print('data shape (raw): ' + str( np_rx.shape ) )
        sys.exit(-1)

      np_t = t.get()
      np_rt = rt.get()
      if np_t.shape != np_rt.shape:
        print('labels shape error (minibatch %d)!' % (idx,))
        print('labels shape: ' + str( np_t.shape ) )
        print('labels shape (raw): ' + str( np_rt.shape ) )
        sys.exit(-1)

      de = numpy.linalg.norm(np_rx - np_x)
      if de != 0.0:
        print( 'Data error: %e (minibatch %d)!' % (de,idx))

      orig_labels = numpy.argmax(np_t, axis=0)
      raw_labels = numpy.argmax(np_rt, axis=0)
      le = numpy.linalg.norm( orig_labels - raw_labels,ord=0 )
      if le != 0:
        print( 'Labels error: %e (minibatch %d)!' % (de,idx))

      if args.graphic:
        np_rx_t = np_rx.transpose()
        np_x_t = np_x.transpose()
        for idx, (xx,rr) in enumerate(zip(np_x_t,np_rx_t)):
          ax1.clear()
          ax2.clear()
          ax1.imshow(xx.reshape(28,28), cmap=plt.cm.Greys)
          ax1.title.set_text( str( orig_labels[idx] ) )
          ax2.imshow(rr.reshape(28,28), cmap=plt.cm.Greys)
          ax2.title.set_text( str( raw_labels[idx] ) )
          plt.draw()
          plt.pause(args.pause)


