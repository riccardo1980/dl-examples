"""A deep MNIST classifier using convolutional layers.

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import time
import math

from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf

import cmdline_args

from mnist_model import deepnn_nn, deepnn_layers

API_MAP = {'nn': deepnn_nn,
           'layers': deepnn_layers}

def main(_):
  """ Main function
  """
  # Import data
  mnist = input_data.read_data_sets(cmdline_args.FLAGS.data_dir, one_hot=True)

  images = tf.placeholder(tf.float32, [None, 784])
  labels = tf.placeholder(tf.float32, [None, 10])
  keep_prob = tf.placeholder(tf.float32)

  # Build the graph for the deep net
  logits = API_MAP[cmdline_args.FLAGS.tf_api](images, keep_prob)

  with tf.name_scope('loss'):
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=labels,
                                                            logits=logits)
  cross_entropy = tf.reduce_mean(cross_entropy)

  with tf.name_scope('adam_optimizer'):
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

  with tf.name_scope('accuracy'):
    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
    correct_prediction = tf.cast(correct_prediction, tf.float32)
  accuracy = tf.reduce_mean(correct_prediction)

  graph_location = cmdline_args.FLAGS.graph_dir
  print('Saving graph to: %s' % graph_location)
  train_writer = tf.summary.FileWriter(graph_location)
  train_writer.add_graph(tf.get_default_graph())

  config = tf.ConfigProto(inter_op_parallelism_threads=cmdline_args.FLAGS.inter_threads,
                          intra_op_parallelism_threads=cmdline_args.FLAGS.intra_threads)

  total_duration = 0.0
  total_duration_squared = 0.0

  with tf.Session(config=config) as sess:
    sess.run(tf.global_variables_initializer())

    if cmdline_args.FLAGS.verbose:
      # print all trainable variabes:
      for v in tf.trainable_variables():
        print("Variable: %s, shape %s " % (v.name, str(v.shape)))

    for ii in range(cmdline_args.FLAGS.maxit):
      batch = mnist.train.next_batch(cmdline_args.FLAGS.batch_size,
                                     shuffle=not cmdline_args.FLAGS.no_batch_shuffle)

      if (cmdline_args.FLAGS.accuracy_interval > 0 and
          ii % cmdline_args.FLAGS.accuracy_interval == 0):
        train_accuracy = accuracy.eval(feed_dict={
            images: batch[0], labels: batch[1], keep_prob: 1.0})
        print('step % 6d, training accuracy %g' % (ii, train_accuracy))

      start_time = time.time()
      train_step.run(feed_dict={images: batch[0], labels: batch[1], keep_prob: 0.5})
      duration = time.time() - start_time

      total_duration += duration
      total_duration_squared += duration * duration

    mn = total_duration / cmdline_args.FLAGS.maxit
    vr = total_duration_squared / cmdline_args.FLAGS.maxit - mn * mn
    sd = math.sqrt(vr)

    test_start_time = time.time()
    acc = accuracy.eval(feed_dict={
        images: mnist.test.images, labels: mnist.test.labels, keep_prob: 1.0})
    test_duration = time.time() - test_start_time

    print('---------------------------------------------------------')
    print('Training time (avg. across %d steps, %d images / batch): %.3f +/- %.3f sec / batch' %
          (cmdline_args.FLAGS.maxit, cmdline_args.FLAGS.batch_size, mn, sd))

    print('Test time (%d images): %.3f sec'  %
          (len(mnist.test.labels), test_duration))

    print('Test accuracy: %g' % acc)

if __name__ == '__main__':

  if cmdline_args.FLAGS.verbose:
    for key, val in vars(cmdline_args.FLAGS).iteritems():
      print('%s: %s' % (key, str(val)))

  tf.app.run(main=main, argv=[sys.argv[0]])
