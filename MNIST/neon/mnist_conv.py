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

from neon.callbacks.callbacks import Callbacks
from neon.data import MNIST
from neon.initializers import Gaussian, Constant
from neon.layers import GeneralizedCost, Affine, Conv, Pooling, Dropout
from neon.models import Model
from neon.optimizers import GradientDescentMomentum
from neon.transforms import Rectlin, Logistic, CrossEntropyMulti, Accuracy, Softmax
from neon.util.argparser import NeonArgparser
from neon import logger as neon_logger


# parse the command line arguments
parser = NeonArgparser(__doc__)

args = parser.parse_args()

# load up the mnist data set
dataset = MNIST(path=args.data_dir)
train_set = dataset.train_iter
valid_set = dataset.valid_iter

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

# run fit
conv.fit(train_set, optimizer=optimizer,
        num_epochs=args.epochs, cost=cost, callbacks=callbacks)
accuracy = conv.eval(valid_set, metric=Accuracy())
neon_logger.display('Accuracy = %.1f%%' % (accuracy * 100))
