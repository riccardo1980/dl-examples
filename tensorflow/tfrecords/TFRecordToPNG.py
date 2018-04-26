#!/usr/bin/env python

from  __future__ import print_function
import argparse
import os
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import numpy as np
import PIL



if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('filename', type=str, help='input file', nargs=1)
    args = parser.parse_args()

    # compression handling
    _, ext = os.path.splitext(args.filename[0])
    extension_to_compression = {
        '.gz': tf.python_io.TFRecordCompressionType.GZIP,
        '.zip': tf.python_io.TFRecordCompressionType.ZLIB,
    }
    if ext in extension_to_compression.keys():
        compression = extension_to_compression[ext]
    else:
        compression = tf.python_io.TFRecordCompressionType.NONE

    # compression options
    reader_opt = tf.python_io.TFRecordOptions(compression)
    reader = tf.TFRecordReader(options=reader_opt)

    filename_queue = tf.train.string_input_producer(args.filename, num_epochs=1)

    _, serialized_example = reader.read(filename_queue)
    feature_map = {
        'height': tf.FixedLenFeature([], tf.int64),
        'width': tf.FixedLenFeature([], tf.int64),
        'depth': tf.FixedLenFeature([], tf.int64),
        'label': tf.FixedLenFeature([], tf.int64),
        'image_raw': tf.FixedLenFeature([], tf.string)
    }
    features = tf.parse_single_example(serialized_example, features=feature_map)
    height = tf.cast(features['height'], tf.int32)
    width = tf.cast(features['width'], tf.int32)
    depth = tf.cast(features['depth'], tf.int32)
    label = tf.cast(features['label'], tf.int32)
    image_raw = tf.reshape(tf.decode_raw(features['image_raw'], tf.uint8), [height, width, depth])

    init_op = tf.group(tf.global_variables_initializer(),
                       tf.local_variables_initializer())

    with tf.Session() as sess:
        sess.run(init_op)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        try:
            while True:
                lbl, image = sess.run([label,image_raw])
                image = np.squeeze(image)
                print('label: {} shape: {} type: {}'.format(lbl, image.shape, image.dtype))
                if image.ndim == 2 and image.dtype == 'uint8':
                    PIL.Image.fromarray(image).save('img1.png', 'png')
                break
        except tf.errors.OutOfRangeError as e:
            coord.request_stop(e)
        finally:
            coord.request_stop()
            coord.join(threads)

