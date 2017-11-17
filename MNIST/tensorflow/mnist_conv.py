"""A deep MNIST classifier using convolutional layers.

"""
# pylint: disable=invalid-name
# pylint: disable=g-bad-import-order

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import tempfile
import time
import math
import os

from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf

FLAGS = None


def deepnn(x):
  """deepnn builds the graph for a deep net for classifying digits.

  Args:
    x: an input tensor with the dimensions (N_examples, 784), where 784 is the
    number of pixels in a standard MNIST image.

  Returns:
    A tuple (y, keep_prob). y is a tensor of shape (N_examples, 10), with values
    equal to the logits of classifying the digit into one of 10 classes (the
    digits 0-9). keep_prob is a scalar placeholder for the probability of
    dropout.
  """
  # Reshape to use within a convolutional neural net.
  # Last dimension is for "features" - there is only one here, since images are
  # grayscale -- it would be 3 for an RGB image, 4 for RGBA, etc.
  with tf.name_scope('reshape'):
    x_image = tf.reshape(x, [-1, 28, 28, 1])

  # First convolutional layer - maps one grayscale image to 32 feature maps.
  with tf.name_scope('conv1'):
    W_conv1 = weight_variable([5, 5, 1, 32])
    b_conv1 = bias_variable([32])
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)

  # Pooling layer - downsamples by 2X.
  with tf.name_scope('pool1'):
    h_pool1 = max_pool_2x2(h_conv1)

  # Second convolutional layer -- maps 32 feature maps to 64.
  with tf.name_scope('conv2'):
    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)

  # Second pooling layer.
  with tf.name_scope('pool2'):
    h_pool2 = max_pool_2x2(h_conv2)

  # Fully connected layer 1 -- after 2 round of downsampling, our 28x28 image
  # is down to 7x7x64 feature maps -- maps this to 1024 features.
  with tf.name_scope('fc1'):
    W_fc1 = weight_variable([7 * 7 * 64, 1024])
    b_fc1 = bias_variable([1024])

    h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

  # Dropout - controls the complexity of the model, prevents co-adaptation of
  # features.
  with tf.name_scope('dropout'):
    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob, seed=FLAGS.dropout_seed)

  # Map the 1024 features to 10 classes, one for each digit
  with tf.name_scope('fc2'):
    W_fc2 = weight_variable([1024, 10])
    b_fc2 = bias_variable([10])

    y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
  return y_conv, keep_prob


def conv2d(x, W):
  """conv2d returns a 2d convolution layer with full stride."""
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
  """max_pool_2x2 downsamples a feature map by 2X."""
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')


def weight_variable(shape):
  """weight_variable generates a weight variable of a given shape."""
  initial = tf.truncated_normal(shape, stddev=0.1, seed=FLAGS.weight_init_seed)
  return tf.Variable(initial)


def bias_variable(shape):
  """bias_variable generates a bias variable of a given shape."""
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)


def main(_):
  # Import data
  mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

  # Create the model
  x = tf.placeholder(tf.float32, [None, 784])

  # Define loss and optimizer
  y_ = tf.placeholder(tf.float32, [None, 10])

  # Build the graph for the deep net
  y_conv, keep_prob = deepnn(x)

  with tf.name_scope('loss'):
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_,
                                                            logits=y_conv)
  cross_entropy = tf.reduce_mean(cross_entropy)

  with tf.name_scope('adam_optimizer'):
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

  with tf.name_scope('accuracy'):
    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    correct_prediction = tf.cast(correct_prediction, tf.float32)
  accuracy = tf.reduce_mean(correct_prediction)

  graph_location = FLAGS.graph_dir 
  print('Saving graph to: %s' % graph_location)
  train_writer = tf.summary.FileWriter(graph_location)
  train_writer.add_graph(tf.get_default_graph())

  config=tf.ConfigProto( inter_op_parallelism_threads=FLAGS.inter_threads,
      intra_op_parallelism_threads=FLAGS.intra_threads)
 
  total_duration = 0.0
  total_duration_squared = 0.0
  
  with tf.Session( config=config ) as sess:
    sess.run(tf.global_variables_initializer())
    for ii in range( FLAGS.maxit ):
      batch = mnist.train.next_batch(FLAGS.batch_size, shuffle= not FLAGS.no_batch_shuffle)

      if FLAGS.accuracy_interval > 0 and ii % FLAGS.accuracy_interval == 0:
        train_accuracy = accuracy.eval(feed_dict={
            x: batch[0], y_: batch[1], keep_prob: 1.0})
        print('step % 6d, training accuracy %g' % (ii, train_accuracy))

      start_time = time.time()
      train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
      duration = time.time() - start_time 
 
      total_duration += duration
      total_duration_squared += duration * duration

    mn = total_duration / FLAGS.maxit
    vr = total_duration_squared / FLAGS.maxit - mn * mn
    sd = math.sqrt(vr)

    test_start_time = time.time()
    acc = accuracy.eval(feed_dict={
        x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0})
    test_duration = time.time() - test_start_time 

    print('---------------------------------------------------------')
    print('Training time (avg. across %d steps, %d images / batch): %.3f +/- %.3f sec / batch' %
        (FLAGS.maxit, FLAGS.batch_size, mn, sd))
 
    print('Test time (%d images): %.3f sec'  %
        (len(mnist.test.labels), test_duration))

    print('Test accuracy: %g' % acc )

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  
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
  parser.add_argument('--verbose', action="store_true",
                      help='Increase verbosity')

  FLAGS, unparsed = parser.parse_known_args()
 
  if FLAGS.verbose:
    for key, val in vars(FLAGS).iteritems():
      print('%s: %s' % (key, str(val)))
 
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
