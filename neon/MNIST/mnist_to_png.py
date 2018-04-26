import logging
import numpy

from neon.util.argparser import NeonArgparser
from neon import logger as neon_logger

from neon.data import MNIST
from ConservativeArrayIterator import ConservativeArrayIterator
from LecunMNIST import LecunMNIST

from scipy.misc import imsave
import numpy as np
import os
"""
Save images in train/validation sets in separate folders

This tool prepares folder trees and tsv files suitable to
be handled by aeon DataLoader

Example:
  python mnist_to_png.py

  tree ./
  ./
  |-- test
  |   |-- 00
  |   |   |-- 000.png
  |   |   |-- :::.png
  |   |   `-- 999.png
  |   |
  :   |-- 01
  :   :   :
  `-- train
      |-- 00
      |   |-- 000.png
      |   |-- :::.png
      |   `-- 999.png
      :

"""
def create_tree( X, y, rootdir, name):
  """Save dataset to folder

  X images
  y labels
  rootdir base folder
  name tsv file name
  """
  # create root folder
  try:
    os.stat(rootdir)
  except:
    os.mkdir(rootdir)

  # create manifest
  filename =  os.path.join(rootdir, name+'.tsv')
  fd = open(filename,'w')
  fd.write('@FILE\tASCII_INT\n')

  for idx, (img, lab) in enumerate(zip(X, y)):
    lev0 = idx % 1000
    lev1 = idx // 1000
    # create second level folder
    tmpdir = os.path.join(rootdir, str(lev1).zfill(2))
    try:
      os.stat(tmpdir)
    except:
      os.mkdir(tmpdir)

    tmpfile = os.path.join(tmpdir, str(lev0).zfill(3))
    imsave (tmpfile+'.png', np.uint8(255.0 * img.reshape((28,28))) )
    fd.write(tmpfile+'.png'+'\t'+str(lab)+'\n')

  fd.close()

if __name__ == '__main__':

  logger = logging.getLogger(__name__)
  # parse the command line arguments
  parser = NeonArgparser(__doc__)

  args = parser.parse_args()
  dataset = LecunMNIST(path=args.data_dir)

  (X_train, y_train), (X_test, y_test), nclass = dataset.load_data()

  create_tree( X_train, y_train, 'train', 'train')
  create_tree( X_train, y_train, 'test', 'test')

