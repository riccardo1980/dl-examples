import argparse
import os
parser = argparse.ArgumentParser()

# TensorFlow api
parser.add_argument('--tf_api', type=str, default='layers',
                    help='TensorFlow api')

# folders
parser.add_argument('--data_dir', type=str,
                    default='/tmp/tensorflow/mnist/input_data',
                    help='Directory for storing input data')
parser.add_argument('--graph_dir', type=str, 
                    default=os.path.join( os.getcwd(), 'graph' ),
                    help='Directory where to store graph file')

# thread pool handling
parser.add_argument('--inter_threads', type=int,
                    default=0, help='Inter op parallelism threads')
parser.add_argument('--intra_threads', type=int,
                    default=0, help='Intra op parallelism threads')

# batch extraction
parser.add_argument('--batch_size', type=int,
                    default=32, help='Batch size')
parser.add_argument('--no_batch_shuffle', action="store_true",
                    help='Prevent dataset shuffle')
parser.add_argument('--maxit', type=int,
                    default=10000, help='Maximum number of iterations')

# accuracy evaluation in training
parser.add_argument('--accuracy_interval', type=int, default=0,
                    help='Step between two accuracy evaluation on test set during training')

# seed settings
parser.add_argument('--batch_seed', type=int, default=None,
                    help='Set seed for deterministic batch extraction')
parser.add_argument('--dropout_seed', type=int, default=None,
                    help='Set seed for deterministic dropout')
parser.add_argument('--weight_init_seed', type=int, default=None,
                    help='Set seed for deterministic weights initialization')

# verbosity level
parser.add_argument('-v','--verbose', action="store_true",
                    help='Increase verbosity')

FLAGS = parser.parse_args()

