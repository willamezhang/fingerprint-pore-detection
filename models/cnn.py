import tensorflow as tf


class CNN:
  def __init__(self, images, patch_size=17, layers=7):
    # kernel size is fixed across entire net
    kernel_size = [3, 3]

    # input of first layer is 'images'
    prev = images

    # conv layers
    for i in range(1, 1 + layers):
      prev = tf.layers.conv2d(
          prev,
          filters=2**((i + 1) // 2 + 5),
          kernel_size=kernel_size,
          activation=tf.nn.relu,
          name='conv{}'.format(i))

    # flatten last conv layer
    last_conv_flat = tf.reshape(prev, [
        -1, 2**
        ((layers + 1) // 2 + 5) * (patch_size - layers * (kernel_size[0] - 1))
        * (patch_size - layers * (kernel_size[1] - 1))
    ])

    # fc + relu layer
    fc1 = tf.layers.dense(
        last_conv_flat, 4096, activation=tf.nn.relu, name='fc1')

    # final fc layer
    self.logits = tf.layers.dense(fc1, 2, name='fc2')

    # build prediction op
    self.predictions = tf.nn.softmax(self.logits)

  def build_loss(self, labels):
    xentropy = tf.nn.softmax_cross_entropy_with_logits_v2(
        labels=labels, logits=self.logits, name='xentropy')
    self.loss = tf.reduce_mean(xentropy, name='xentropy_mean')

    return self.loss

  def build_train(self, learning_rate):
    global_step = tf.Variable(1, name='global_step', trainable=False)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    self.train = optimizer.minimize(self.loss, global_step=global_step)

    return self.train
