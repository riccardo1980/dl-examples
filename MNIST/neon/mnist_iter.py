import logging
import numpy

from neon.util.argparser import NeonArgparser
from neon import logger as neon_logger

from neon.data import MNIST
from LecunMNIST import LecunMNIST
from ConservativeArrayIterator import ConservativeArrayIterator
from neon.data.dataiterator import ArrayIterator

if __name__ == '__main__':

  logger = logging.getLogger(__name__)
  # parse the command line arguments
  parser = NeonArgparser(__doc__)

  args = parser.parse_args()
  #dataset = LecunMNIST(path=args.data_dir)
  dataset = MNIST(path=args.data_dir)

  (X_train, y_train), (X_test, y_test), nclass = dataset.load_data()

  """
  # take only first n
  n = 125
  X_train = X_train[:n,]
  y_train = y_train[:n]
  """
  train = ConservativeArrayIterator(X_train,
                                    y_train,
                                    nclass=nclass,
                                    lshape=(1, 28, 28),
                                    name='train')
  val = None
  val = ConservativeArrayIterator(X_test,
                                  y_test,
                                  nclass=nclass,
                                  lshape=(1, 28, 28),
                                  name='valid')
  _data_dict = {'train': train,
                'valid': val}

  print("Elements: %g each one of %g" % (len(X_train), len(X_train[0])))
  for e in range(args.epochs):
    print("Epoch: %g, batches %g" % (e,train.nbatches) )
    for imgs, labs in train:
      np_imgs = imgs[:,0].get().flatten()
      print( np_imgs[np_imgs.nonzero()][:3] )

