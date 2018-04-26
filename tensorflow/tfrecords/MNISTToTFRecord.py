#!/usr/bin/env python

"""Converts MNIST data to TFRecords file format with Example protos."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys

import tensorflow as tf

from tensorflow.contrib.learn.python.learn.datasets import mnist

FLAGS = None

def _int64_feature(value):
  """Creates an int64 list feature from values"""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
  """Creates a bytes list feature from values"""
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def convert_to(data_set, filename):
  """Converts a dataset to tfrecords
        data_set    must contain images and labels fields
        filename    output filename
                    {.gz, .zip} extensions trigger {gzip, zlib} compressions
                    any other extension triggers no compression
  """

  # image size handling
  images = data_set.images
  labels = data_set.labels
  num_examples = data_set.num_examples
  if images.shape[0] != num_examples:
    raise ValueError('Images size %d does not match label size %d.' %
                     (images.shape[0], num_examples))
  rows = images.shape[1]
  cols = images.shape[2]
  depth = images.shape[3]

  # compression handling
  _, ext = os.path.splitext(filename)
  extension_to_compression = {
    '.gz': tf.python_io.TFRecordCompressionType.GZIP,
    '.zip': tf.python_io.TFRecordCompressionType.ZLIB,
  }
  if ext in extension_to_compression.keys():
    compression = extension_to_compression[ext]
  else:
    compression = tf.python_io.TFRecordCompressionType.NONE

  # writing
  print('Writing', filename)
  writer_opt = tf.python_io.TFRecordOptions(compression)
  with tf.python_io.TFRecordWriter(filename, options=writer_opt) as writer:
    for index,(img,lbl) in enumerate(zip(images,labels)):
      image_raw = img.tostring()
      example = tf.train.Example(
          features=tf.train.Features(
              feature={
                  'height': _int64_feature(rows),
                  'width': _int64_feature(cols),
                  'depth': _int64_feature(depth),
                  'label': _int64_feature(int(lbl)),
                  'image_raw': _bytes_feature(image_raw)
              }))
      writer.write(example.SerializeToString())


def main(unused_argv):
  # Get the data.
  data_sets = mnist.read_data_sets(FLAGS.data_path,
                                   dtype=tf.uint8,
                                   reshape=False,
                                   validation_size=FLAGS.validation_size)

  # used to infer extension to requested compression
  extension_map = {
    'gzip': '.gz',
    'zlib': '.zip',
    'none': ''}

  # Convert to Examples and write the result to TFRecords.
  if not os.path.exists(FLAGS.tf_path):
    os.makedirs(FLAGS.tf_path)

  convert_to(data_sets.train,
             os.path.join(FLAGS.tf_path, 'train.tfrecords'+extension_map[FLAGS.compression]))
  convert_to(data_sets.validation,
             os.path.join(FLAGS.tf_path, 'validation.tfrecords'+extension_map[FLAGS.compression]))
  convert_to(data_sets.test,
             os.path.join(FLAGS.tf_path, 'test.tfrecords'+extension_map[FLAGS.compression]))

if __name__ == '__main__':
  parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument(
      '--data_path',
      type=str,
      default='/tmp/data',
      help='Directory to download data files and write the converted result'
  )
  parser.add_argument(
      '--tf_path',
      type=str,
      default='/tmp/data',
      help='Directory to write the converted result'
  )
  parser.add_argument(
      '--validation_size',
      type=int,
      default=5000,
      help="""\
      Number of examples to separate from the training data for the validation
      set.\
      """
  )
  parser.add_argument('--compression', default='gzip', choices=['none', 'gzip', 'zlib'], help='Required compression')

  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
