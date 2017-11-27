

import tensorflow as tf

import cmdline_args

def deepnn_layers(images, keep_prob):
  """deepnn builds the graph for a deep net for classifying digits.

  Args:
    images: an input tensor with the dimensions (N_examples, 784), where 784 is the
    number of pixels in a standard MNIST image.
    keep_prob: a scalar placeholder for the probability of
    dropout.

  Returns:
    y: a tensor of shape (N_examples, 10), with values
    equal to the logits of classifying the digit into one of 10 classes (the
    digits 0-9).
  """
  # Reshape to use within a convolutional neural net.
  # Last dimension is for "features" - there is only one here, since images are
  # grayscale -- it would be 3 for an RGB image, 4 for RGBA, etc.
  with tf.name_scope('reshape'):
    x_image = tf.reshape(images, [-1, 28, 28, 1])

  truncated_normal_init = tf.truncated_normal_initializer(stddev=0.1,
                                                          seed=cmdline_args.FLAGS.weight_init_seed)
  # First convolutional layer - maps one grayscale image to 32 feature maps.
  h_conv1 = tf.layers.conv2d(x_image, 32, [5, 5],
                             kernel_initializer=truncated_normal_init,
                             bias_initializer=tf.constant_initializer(0.1),
                             padding="same", activation=tf.nn.relu, name='conv1')

  # Pooling layer - downsamples by 2X.
  h_pool1 = tf.layers.max_pooling2d(h_conv1, [2, 2], 2, name='pool1')

  # Second convolutional layer -- maps 32 feature maps to 64.
  h_conv2 = tf.layers.conv2d(h_pool1, 64, [5, 5],
                             kernel_initializer=truncated_normal_init,
                             bias_initializer=tf.constant_initializer(0.1),
                             padding="same", activation=tf.nn.relu, name='conv2')

  # Second pooling layer.
  h_pool2 = tf.layers.max_pooling2d(h_conv2, [2, 2], 2, name='pool2')

  # Fully connected layer 1 -- after 2 round of downsampling, our 28x28 image
  # is down to 7x7x64 feature maps -- maps this to 1024 features.
  # it will be: h_pool2_flat = tf.layers.flatten( h_pool2, name='flatten' )
  h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
  h_fc1 = tf.layers.dense(h_pool2_flat, 1024, activation=tf.nn.relu,
                          kernel_initializer=truncated_normal_init,
                          bias_initializer=tf.constant_initializer(0.1),
                          name='fc1')

  # Dropout - controls the complexity of the model, prevents co-adaptation of
  # features.
  with tf.name_scope('dropout'):
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob, seed=cmdline_args.FLAGS.dropout_seed)

  # Map the 1024 features to 10 classes, one for each digit
  logits = tf.layers.dense(h_fc1_drop, 10,
                           kernel_initializer=truncated_normal_init,
                           bias_initializer=tf.constant_initializer(0.1),
                           name='fc2')

  return logits


def deepnn_nn(images, keep_prob):
  """deepnn builds the graph for a deep net for classifying digits.

  Args:
    images: an input tensor with the dimensions (N_examples, 784), where 784 is the
    number of pixels in a standard MNIST image.
    keep_prob: a scalar placeholder for the probability of
    dropout.

  Returns:
    y: a tensor of shape (N_examples, 10), with values
    equal to the logits of classifying the digit into one of 10 classes (the
    digits 0-9).
  """
  # Reshape to use within a convolutional neural net.
  # Last dimension is for "features" - there is only one here, since images are
  # grayscale -- it would be 3 for an RGB image, 4 for RGBA, etc.
  with tf.name_scope('reshape'):
    x_image = tf.reshape(images, [-1, 28, 28, 1])

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
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob, seed=cmdline_args.FLAGS.dropout_seed)

  # Map the 1024 features to 10 classes, one for each digit
  with tf.name_scope('fc2'):
    W_fc2 = weight_variable([1024, 10])
    b_fc2 = bias_variable([10])

    logits = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
  return logits

def conv2d(images, W):
  """conv2d returns a 2d convolution layer with full stride."""
  return tf.nn.conv2d(images, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(images):
  """max_pool_2x2 downsamples a feature map by 2X."""
  return tf.nn.max_pool(images, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')


def weight_variable(shape, name='kernel'):
  """weight_variable generates a weight variable of a given shape."""
  initial = tf.truncated_normal(shape, stddev=0.1, seed=cmdline_args.FLAGS.weight_init_seed)
  return tf.Variable(initial, name=name)


def bias_variable(shape, name='bias'):
  """bias_variable generates a bias variable of a given shape."""
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial, name=name)
