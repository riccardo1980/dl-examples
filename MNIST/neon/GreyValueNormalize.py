from neon.data.dataloader_transformers import ValueNormalize

class GreyValueNormalize(ValueNormalize):
  """Gray level image normalizer
  """
  def transform(self, t):
    # create a view of t and modify that.  Modifying t directly doesn't
    # work for some reason ...
    tr = t.reshape((1, -1))
    tr[:] = (tr - self.xmin) / self.xspan * self.yspan + self.ymin
    return t

