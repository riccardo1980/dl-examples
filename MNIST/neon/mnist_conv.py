"""A deep MNIST classifier using convolutional layers.

Examples:

    python mnist_conv.py -b gpu -e 10

        Run the example for 10 epochs using the NervanaGPU backend

    python mnist_conv.py --eval_freq 1

        After each training epoch, process the validation/test data
        set through the model and display the cost.

    python mnist_conv.py --serialize 1 -s checkpoint.pkl

        After every iteration of training, dump the model to a pickle
        file named "checkpoint.pkl".  Changing the serialize parameter
        changes the frequency at which the model is saved.

    python mnist_conv.py --model_file checkpoint.pkl

        Before starting to train the model, set the model state to
        the values stored in the checkpoint file named checkpoint.pkl.

"""

from neon.callbacks.callbacks import Callbacks, Callback
from neon.initializers import Gaussian, Constant
from neon.layers import GeneralizedCost, Affine, Conv, Pooling, Dropout
from neon.models import Model
from neon.optimizers import GradientDescentMomentum
from neon.transforms import Rectlin, Logistic, CrossEntropyMulti, Accuracy, Softmax
from neon.util.argparser import NeonArgparser
from neon import logger as neon_logger

from LecunMNIST import LecunMNIST
from ConservativeArrayIterator import ConservativeArrayIterator

class PrintBatchNumber(Callback):
  """
  Simple callback to print train set size, as seen at epoch start
  """
  def on_epoch_end(self, callback_data, model, epoch):
    """
    Print at epoch start
    """
    print('Train set size (epoch %g): %g ' % (epoch, model.nbatches))


if __name__ == '__main__':
  """
  Main function

  - argument parsiong
  - dataset load
  - iterators creation
  - nn definiton
  - cost function definition
  - optimizer definition
  - learning phase
  - accuracy evaluation
  """
  # parse the command line arguments
  parser = NeonArgparser(__doc__)

  args = parser.parse_args()

  """
  # load up the mnist data set
  dataset = LecunMNIST(path=args.data_dir)
  (X_train, y_train), (X_test, y_test), nclass = dataset.load_data()

  train_set = ConservativeArrayIterator(X_train,
                                        y_train,
                                        nclass=nclass,
                                        lshape=(1, 28, 28),
                                        name='train')
  valid_set = ConservativeArrayIterator(X_test,
                                        y_test,
                                        nclass=nclass,
                                        lshape=(1, 28, 28),
                                        name='valid')
  """
  from neon.data.dataloaderadapter import DataLoaderAdapter
  from neon.data.dataloader_transformers import TypeCast, OneHot
  from aeon import DataLoader
  from GreyValueNormalize import GreyValueNormalize
  import numpy as np

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

  # setup weight initialization function
  init_norm = Gaussian(loc=0.0, scale=0.1)

  relu=Rectlin()

  conv1 = dict(init=init_norm, batch_norm=False,
          activation=relu, bias=Constant(0.1))

  # setup model layers
  layers = [Conv((5,5,32), padding=2, **conv1),
            Pooling(2, strides=2),
            Conv((5,5,64), padding=2, **conv1),
            Pooling(2, strides=2),
            Affine(nout=1024, init=init_norm, activation=relu),
            Dropout(0.5),
            Affine(nout=10, init=init_norm, activation=Softmax())]

  # setup cost function as CrossEntropy
  cost = GeneralizedCost(costfunc=CrossEntropyMulti())

  # setup optimizer
  optimizer = GradientDescentMomentum(
      0.1, momentum_coef=0.9, stochastic_round=args.rounding)

  # initialize model object
  conv = Model(layers=layers)

  # configure callbacks
  callbacks = Callbacks(conv, eval_set=valid_set, **args.callback_args)
  callbacks.add_callback(PrintBatchNumber(epoch_freq=1))

  """
  import numpy as np
  for (x,l) in train_set:
    print(np.argmax(l.get(), axis=0))
    print(x.get()[:,0])
    break
  """
  # run fit
  conv.fit(train_set, optimizer=optimizer,
          num_epochs=args.epochs, cost=cost, callbacks=callbacks)

  accuracy = conv.eval(valid_set, metric=Accuracy())
  neon_logger.display('Accuracy = %.1f%%' % (accuracy * 100))
