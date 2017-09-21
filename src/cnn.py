import tensorflow as tf
import numpy as np


class ConvolutionalNeuralNetwork:

    def __init__(
        self,
        BATCH_SIZE=10,
        IMAGE_SIZE=28,
        NUM_LABELS=10,
        NUM_CHANNELS=1,
        NUM_VALIDATION=1000,
        NUM_TEST=10000,
        SEED=42):

        self.BATCH_SIZE = BATCH_SIZE
        self.IMAGE_SIZE = IMAGE_SIZE
        self.NUM_LABELS = NUM_LABELS
        self.NUM_CHANNELS = NUM_CHANNELS
        self.NUM_VALIDATION = NUM_VALIDATION
        self.NUM_TEST = NUM_TEST
        self.SEED = SEED

        self.sess = tf.InteractiveSession()

        self.sess.as_default()

        self.init_vars()
        self.init_model_architecture()
        self.init_training_graph()

    def init_vars(self):
        self.input_node = tf.placeholder(
            tf.float32,
            shape=(None, self.IMAGE_SIZE, self.IMAGE_SIZE, self.NUM_CHANNELS))

        self.train_labels_node = tf.placeholder(tf.float32,
            shape=(self.BATCH_SIZE, self.NUM_LABELS))

        self.conv1_weights = tf.Variable(
        tf.truncated_normal([5, 5, self.NUM_CHANNELS, 32],  # 5x5 filter, depth 32.
                              stddev=0.1,
                              seed=self.SEED))

        self.conv1_biases = tf.Variable(tf.zeros([32]))

        self.conv2_weights = tf.Variable(
            tf.truncated_normal([5, 5, 32, 64],  # truncates samples > 2std
                              stddev=0.1,
                              seed=self.SEED))

        self.conv2_biases = tf.Variable(tf.constant(0.1, shape=[64]))

        self.fc1_weights = tf.Variable(  # fully connected, depth 512.
            tf.truncated_normal(
                [self.IMAGE_SIZE // 4 * self.IMAGE_SIZE // 4 * 64, 512],
                stddev=0.1, seed=self.SEED))

        self.fc1_biases = tf.Variable(tf.constant(0.1, shape=[512]))

        self.fc2_weights = tf.Variable(
            tf.truncated_normal([512, self.NUM_LABELS],
                              stddev=0.1,
                              seed=self.SEED))

        self.fc2_biases = tf.Variable(tf.constant(0.1, shape=[self.NUM_LABELS]))

        self.keep_prob = tf.Variable(0.5)

        print('Model variables initialised')

    def init_model_architecture(self):
        self.conv1 = tf.nn.conv2d(self.input_node, self.conv1_weights,
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

        dropout = tf.nn.dropout(self.hidden1, self.keep_prob, seed=self.SEED)
        self.logits = tf.matmul(dropout, self.fc2_weights) + self.fc2_biases

        print('Model architecture initialised')

    def init_training_graph(self):
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
          logits=self.logits, labels=self.train_labels_node))

        self.regularizers = (
            tf.nn.l2_loss(self.fc1_weights) + tf.nn.l2_loss(self.fc1_biases) +
            tf.nn.l2_loss(self.fc2_weights) + tf.nn.l2_loss(self.fc2_biases))

        self.loss += 5e-4 * self.regularizers

        self.batch = tf.Variable(0)

        self.train_prediction = tf.nn.softmax(self.logits)

        predictions = tf.matmul(self.hidden1, self.fc2_weights) + self.fc2_biases
        self.validation_prediction = tf.nn.softmax(predictions)
        self.test_prediction = tf.nn.softmax(predictions)

        print('Computational graph initialised')

    def train(self, Xtr, Ytr, Xval, Yval):
        self.NUM_LABELS = 10

        num_training = Xtr.shape[0]

        self.learning_rate = tf.train.exponential_decay(
            0.01,
            self.batch * self.BATCH_SIZE,
            num_training,
            0.95,
            staircase=True)

        self.optimizer = tf.train.MomentumOptimizer(
            self.learning_rate, 0.9).minimize(self.loss, global_step=self.batch)

        tf.global_variables_initializer().run()

        summary_writer = tf.summary.FileWriter('logs', graph=self.sess.graph)

        steps = 2000

        for step in range(steps):
            offset = (step * self.BATCH_SIZE) % (num_training - self.BATCH_SIZE)
            batch_data = Xtr[offset:(offset + self.BATCH_SIZE), :, :, :]
            one_hot_encoding = (
                np.arange(self.NUM_LABELS) == Ytr[:, None]).astype(np.float32)
            batch_labels = one_hot_encoding[offset:(offset + self.BATCH_SIZE)]
            feed_dict = {self.input_node: batch_data,
                         self.train_labels_node: batch_labels}

            _, l, lr, predictions = self.sess.run(
              [self.optimizer, self.loss, self.learning_rate, self.train_prediction],
              feed_dict=feed_dict)

            if step % 100 == 0:                
                error = self.error_rate(predictions, batch_labels)
                print('Step %d of %d' % (step, steps))
                print('Mini-batch loss: %.5f Error: %.5f Learning rate: %.5f' % 
                    (l, error, lr))
                print(
                    'Validation error: %.1f%%' % self.error_rate(
                        self.validation_prediction.eval(
                            feed_dict={self.input_node : Xval}),
                        (np.arange(self.NUM_LABELS) == Yval[:, None]).astype(np.float32)))

    def getActivations(self, layer, stimuli):
        units = self.sess.run(layer, feed_dict=
            {self.input_node : stimuli, self.keep_prob : 1.0})
        return units

    def test_model(self, Xte):
        logits = tf.matmul(self.hidden1, self.fc2_weights) + self.fc2_biases
        predictions = np.argmax(
            self.test_prediction.eval(
                feed_dict={self.input_node : Xte}), axis=1).astype(np.int8)
        return predictions

    def error_rate(self, predictions, labels):
        correct = np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
        total = predictions.shape[0]

        error = 100 - (100 * float(correct) / float(total))
        return error
