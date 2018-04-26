#!/usr/bin/env python

from  __future__ import print_function
import argparse
import os
import tensorflow as tf

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('filename', type=str, help='input file', nargs=1)
    args = parser.parse_args()

    filename = args.filename[0]
    _, ext = os.path.splitext(filename)
    extension_to_compression = {
        '.gz': tf.python_io.TFRecordCompressionType.GZIP,
        '.zip': tf.python_io.TFRecordCompressionType.ZLIB,
    }
    if ext in extension_to_compression.keys():
        compression = extension_to_compression[ext]
    else:
        compression = tf.python_io.TFRecordCompressionType.NONE

    # options
    reader_opt = tf.python_io.TFRecordOptions(compression)

    for example in tf.python_io.tf_record_iterator(filename, options=reader_opt):
        result = tf.train.Example.FromString(example)
        print(result)
        break
