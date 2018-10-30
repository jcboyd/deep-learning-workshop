from __future__ import print_function
from __future__ import division

import tensorflow as tf


class ConvolutionalNeuralNetwork:

    def __init__(self, img_size, nb_channels, nb_labels, seed=42):

        self.img_size = img_size
        self.nb_channels = nb_channels
        self.nb_labels = nb_labels
        
        self.seed = seed

        self.sess = tf.InteractiveSession()
        self.sess.as_default()

        self.init_vars()
        self.init_model_architecture()

    def init_vars(self):
        self.X = tf.placeholder(tf.float32,
            shape=(None, self.img_size, self.img_size, self.nb_channels))

        self.Y = tf.placeholder(tf.float32,
            shape=(None, self.nb_labels))

        self.conv1_weights = tf.Variable(
        tf.truncated_normal([5, 5, self.nb_channels, 32],  # 5x5 filter, depth 32.
                              stddev=0.1,
                              seed=self.seed), name='conv1_weights')

        self.conv1_biases = tf.Variable(tf.zeros([32]), name='conv1_biases')

        self.conv2_weights = tf.Variable(
            tf.truncated_normal([5, 5, 32, 64],  # truncates samples > 2std
                              stddev=0.1,
                              seed=self.seed), name='conv2_weights')

        self.conv2_biases = tf.Variable(tf.zeros([64]), name='conv2_biases')

        self.fc1_weights = tf.Variable(  # fully connected, depth 512.
            tf.truncated_normal(
                [self.img_size // 4 * self.img_size // 4 * 64, 512],
                stddev=0.1, seed=self.seed), name='fc1_weights')

        self.fc1_biases = tf.Variable(tf.zeros([512]), name='fc1_biases')

        self.fc2_weights = tf.Variable(
            tf.truncated_normal([512, self.nb_labels],
                              stddev=0.1,
                              seed=self.seed), name='fc2_weights')

        self.fc2_biases = tf.Variable(tf.zeros([self.nb_labels]),
            name='fc2_biases')

        self.keep_prob = tf.Variable(0.5)

        print('Model variables initialised')

    def init_model_architecture(self):
        self.conv1 = tf.nn.conv2d(self.X, self.conv1_weights,
            strides=[1, 1, 1, 1], padding='SAME')

        self.relu1 = tf.nn.relu(tf.nn.bias_add(self.conv1, self.conv1_biases))

        self.pool1 = tf.nn.max_pool(self.relu1,
                              ksize=[1, 2, 2, 1],  # batch size x w x h x c
                              strides=[1, 2, 2, 1],
                              padding='SAME')

        self.conv2 = tf.nn.conv2d(self.pool1,
                            self.conv2_weights,
                            strides=[1, 1, 1, 1],
                            padding='SAME')

        self.relu2 = tf.nn.relu(tf.nn.bias_add(self.conv2, self.conv2_biases))

        self.pool2 = tf.nn.max_pool(self.relu2,
                              ksize=[1, 2, 2, 1],
                              strides=[1, 2, 2, 1],
                              padding='SAME')

        pool_shape = tf.shape(self.pool2)
        self.reshape = tf.reshape(self.pool2,
            [pool_shape[0], pool_shape[1] * pool_shape[2] * pool_shape[3]])

        self.hidden1 = tf.nn.relu(
            tf.matmul(self.reshape, self.fc1_weights) + self.fc1_biases)

        dropout = tf.nn.dropout(self.hidden1, self.keep_prob, seed=self.seed)
        self.logits = tf.matmul(dropout, self.fc2_weights) + self.fc2_biases

        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
          logits=self.logits, labels=self.Y))

        self.regularizers = (
            tf.nn.l2_loss(self.fc1_weights) + tf.nn.l2_loss(self.fc1_biases) +
            tf.nn.l2_loss(self.fc2_weights) + tf.nn.l2_loss(self.fc2_biases))

        self.loss += 5e-4 * self.regularizers

        self.pred = tf.nn.softmax(self.logits)

        print('Model architecture initialised')
