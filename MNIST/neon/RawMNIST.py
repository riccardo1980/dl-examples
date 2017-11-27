import struct
import gzip
import numpy
import os

from neon.data import MNIST
from neon.data.dataiterator import ArrayIterator
from neon.util.argparser import NeonArgparser

def read_images(filepath, normalize=True, sym_range=False ):
    with gzip.open(filepath, 'rb') as f:
        magic = struct.unpack('>i', f.read(4))[0]
        if magic != 2051:
            raise ValueError('Invalid magic number %d in MNIST image file: %s' %
                    (magic, filepath))
        images = struct.unpack('>i', f.read(4))[0]
        rows = struct.unpack('>i', f.read(4))[0]
        cols = struct.unpack('>i', f.read(4))[0]

        buf = f.read( rows * cols * images )
        data = numpy.frombuffer(buf,
                dtype=numpy.uint8).astype(numpy.float32)

        if normalize:
            data = data / 255
            if sym_range:
                data = data *2. -1.
   
        return data.reshape( images, rows * cols * 1)

def read_labels(filepath ):
    with gzip.open(filepath, 'rb') as f:
        magic = struct.unpack('>i', f.read(4))[0]
        if magic != 2049:
            raise ValueError('Invalid magic number %d in MNIST label file: %s' %
                    (magic, filepath))
        labels = struct.unpack('>i', f.read(4))[0]
        buf = f.read( labels )
        data = numpy.frombuffer(buf, dtype=numpy.uint8)
        
        return data

class RawMNIST(MNIST):
    def load_data(self):
        url = 'http://yann.lecun.com/exdb/mnist'
        (X_train, y_train) = self._get_mnist( url, 
            ['train-images-idx3-ubyte.gz','train-labels-idx1-ubyte.gz'],
            [ 9912422, 28881 ], path=self.path,
            normalize=self.normalize, sym_range=self.sym_range)
        (X_test, y_test) = self._get_mnist( url, 
            ['t10k-images-idx3-ubyte.gz','t10k-labels-idx1-ubyte.gz'],
            [1648877, 4542], path=self.path,
            normalize=self.normalize, sym_range=self.sym_range)
        return (X_train, y_train), (X_test, y_test), 10

    def _get_mnist(self, url, files, sizes, path='./', normalize=True, 
         sym_range=False, name='dataset'):

        # images
        filepath = self._valid_path_append(path, files[0])
        if not os.path.exists(filepath):
            self.fetch_dataset(url, files[0], filepath, sizes[0])
        X = read_images( filepath, normalize=normalize,
                sym_range=sym_range )

        # labels
        filepath = self._valid_path_append(path, files[1])
        if not os.path.exists(filepath):
            self.fetch_dataset(url, files[1], filepath, sizes[1])
        y = read_labels( filepath )

        return (X, y)


