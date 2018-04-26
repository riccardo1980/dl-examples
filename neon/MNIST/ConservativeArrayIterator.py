import numpy

from neon.data.dataiterator import ArrayIterator

class ConservativeArrayIterator(ArrayIterator):
  @property
  def nbatches(self):
    """
    Return the number of minibatches in this dataset.
    """
    return self.ndata // self.be.bsz
  def __iter__(self):
    """
    Returns a new minibatch of data with each call.

    The number of minibatches (constant at each epoch) is obtained as the
    maximum of minibatches in which the dataset can be splitted into.

    Each item in the dataset is used at most once in each epoch.

    Yields:
        tuple: The next minibatch which includes both features and labels.
    """
    i1 = self.start
    for bb in range(self.nbatches):
      bsz = min(self.be.bsz, self.ndata - i1)
      islice1, oslice1 = slice(0, bsz), slice(i1, i1 + bsz)
      islice2, oslice2 = None, None

      self.start = i1 + self.be.bsz
      i1 = i1 + self.be.bsz

      if self.be.bsz > bsz:
        islice2, oslice2 = slice(bsz, None), slice(0, self.be.bsz - bsz)
        self.start = self.be.bsz - bsz
        i1 = self.start

      for buf, dev, unpack_func in zip(self.hbuf, self.dbuf, self.unpack_func):
        unpack_func(dev[oslice1], buf[:, islice1])
        if oslice2:
          unpack_func(dev[oslice2], buf[:, islice2])

      inputs = self.Xbuf[0] if len(self.Xbuf) == 1 else self.Xbuf
      targets = self.ybuf if self.ybuf else inputs
      yield (inputs, targets)

