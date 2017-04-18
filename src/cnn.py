import tensorflow as tf
import numpy as np


# We'll bundle groups of examples during training for efficiency.
# This defines the size of the batch.
BATCH_SIZE = 10
IMAGE_SIZE = 28
NUM_LABELS = 10
# We have only one channel in our grayscale images.
NUM_CHANNELS = 1
# The random seed that defines initialization.
SEED = 42


class ConvolutionalNeuralNetwork:

    def __init__(self, Xval, Xte):
        # This is where training samples and labels are fed to the graph.
        # These placeholder nodes will be fed a batch of training data at each
        # training step, which we'll write once we define the graph structure.
        self.train_data_node = tf.placeholder(
            tf.float32,
            shape=(BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS))

        self.train_labels_node = tf.placeholder(tf.float32,
            shape=(BATCH_SIZE, NUM_LABELS))

        # For the validation and test data, we'll just hold the entire dataset in
        # one constant node.
        self.validation_data_node = tf.constant(Xval)
        self.test_data_node = tf.constant(Xte)

        # The variables below hold all the trainable weights. For each, the
        # parameter defines how the variables will be initialized.
        self.conv1_weights = tf.Variable(
        tf.truncated_normal([5, 5, NUM_CHANNELS, 32],  # 5x5 filter, depth 32.
                              stddev=0.1,
                              seed=SEED))

        self.conv1_biases = tf.Variable(tf.zeros([32]))

        self.conv2_weights = tf.Variable(
            tf.truncated_normal([5, 5, 32, 64],
                              stddev=0.1,
                              seed=SEED))

        self.conv2_biases = tf.Variable(tf.constant(0.1, shape=[64]))

        self.fc1_weights = tf.Variable(  # fully connected, depth 512.
            tf.truncated_normal([IMAGE_SIZE // 4 * IMAGE_SIZE // 4 * 64, 512],
                              stddev=0.1,
                              seed=SEED))

        self.fc1_biases = tf.Variable(tf.constant(0.1, shape=[512]))

        self.fc2_weights = tf.Variable(
            tf.truncated_normal([512, NUM_LABELS],
                              stddev=0.1,
                              seed=SEED))

        self.fc2_biases = tf.Variable(tf.constant(0.1, shape=[NUM_LABELS]))

        # Create a new interactive session that we'll use in
        # subsequent code cells.
        self.sess = tf.InteractiveSession()

        # Use our newly created session as the default for 
        # subsequent operations.
        self.sess.as_default()

    def model_architecture(self, data, train=False):
        """The Model definition."""
        # 2D convolution, with 'SAME' padding (i.e. the output feature map has
        # the same size as the input). Note that {strides} is a 4D array whose
        # shape matches the data layout: [image index, y, x, depth].
        conv = tf.nn.conv2d(data,
                            self.conv1_weights,
                            strides=[1, 1, 1, 1],
                            padding='SAME')

        # Bias and rectified linear non-linearity.
        relu = tf.nn.relu(tf.nn.bias_add(conv, self.conv1_biases))

        # Max pooling. The kernel size spec ksize also follows the layout of
        # the data. Here we have a pooling window of 2, and a stride of 2.
        pool = tf.nn.max_pool(relu,
                              ksize=[1, 2, 2, 1],
                              strides=[1, 2, 2, 1],
                              padding='SAME')

        conv = tf.nn.conv2d(pool,
                            self.conv2_weights,
                            strides=[1, 1, 1, 1],
                            padding='SAME')

        relu = tf.nn.relu(tf.nn.bias_add(conv, self.conv2_biases))

        pool = tf.nn.max_pool(relu,
                              ksize=[1, 2, 2, 1],
                              strides=[1, 2, 2, 1],
                              padding='SAME')

        # Reshape the feature map cuboid into a 2D matrix to feed it to the
        # fully connected layers.
        pool_shape = pool.get_shape().as_list()
        reshape = tf.reshape(
            pool,
            [pool_shape[0], pool_shape[1] * pool_shape[2] * pool_shape[3]])

        # Fully connected layer. Note that the '+' operation automatically
        # broadcasts the biases.
        hidden = tf.nn.relu(tf.matmul(reshape, self.fc1_weights) + self.fc1_biases)

        # Add a 50% dropout during training only. Dropout also scales
        # activations such that no rescaling is needed at evaluation time.
        if train:
            hidden = tf.nn.dropout(hidden, 0.5, seed=SEED)
        return tf.matmul(hidden, self.fc2_weights) + self.fc2_biases

        print('Done')

    def init_training_graph(self, num_training):
        # Training computation: logits + cross-entropy loss.
        self.logits = self.model_architecture(self.train_data_node, True)
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
          logits=self.logits, labels=self.train_labels_node))

        # L2 regularization for the fully connected parameters.
        self.regularizers = (tf.nn.l2_loss(self.fc1_weights) + tf.nn.l2_loss(self.fc1_biases) +
                        tf.nn.l2_loss(self.fc2_weights) + tf.nn.l2_loss(self.fc2_biases))
        # Add the regularization term to the loss.
        self.loss += 5e-4 * self.regularizers

        # Optimizer: set up a variable that's incremented once per batch and
        # controls the learning rate decay.
        self.batch = tf.Variable(0)
        # Decay once per epoch, using an exponential schedule starting at 0.01.
        self.learning_rate = tf.train.exponential_decay(
          0.01,                # Base learning rate.
          self.batch * BATCH_SIZE,  # Current index into the dataset.
          num_training,        # Decay step.
          0.95,                # Decay rate.
          staircase=True)
        # Use simple momentum for the optimization.
        self.optimizer = tf.train.MomentumOptimizer(self.learning_rate,
                                               0.9).minimize(self.loss,
                                                             global_step=self.batch)

        # Predictions for the minibatch, validation set and test set.
        self.train_prediction = tf.nn.softmax(self.logits)
        # We'll compute them only once in a while by calling their {eval()} method.
        self.validation_prediction = tf.nn.softmax(self.model_architecture(self.validation_data_node))
        self.test_prediction = tf.nn.softmax(self.model_architecture(self.test_data_node))

        # Initialize all the variables we defined above.
        tf.global_variables_initializer().run()

        print('Done')

    def train_iteration(self, batch_data, batch_labels):
        # This dictionary maps the batch data (as a numpy array) to the
        # node in the graph it should be fed to.
        feed_dict = {self.train_data_node: batch_data,
                     self.train_labels_node: batch_labels}

        # Run the graph and fetch some of the nodes.
        _, l, lr, predictions = self.sess.run([self.optimizer, self.loss, self.learning_rate, self.train_prediction], feed_dict=feed_dict)

        return l, lr, predictions

    def test_model(self):
        predictions = np.argmax(self.test_prediction.eval(), axis=1).astype(np.int8)
        return predictions

    def error_rate(self, predictions, labels):
        """Return the error rate and confusions."""
        correct = np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
        total = predictions.shape[0]

        error = 100.0 - (100 * float(correct) / float(total))

        confusions = np.zeros([10, 10], np.float32)
        bundled = zip(np.argmax(predictions, 1), np.argmax(labels, 1))
        for predicted, actual in bundled:
            confusions[predicted, actual] += 1

        return error, confusions

    def train(self, Xtr, Ytr, Yval):
        BATCH_SIZE = 10
        NUM_LABELS = 10

        num_training = Xtr.shape[0]

        # Train over the first 1/4th of our training set.
        steps = num_training // BATCH_SIZE
        for step in range(steps):
            # Compute the offset of the current minibatch in the data.
            # Note that we could use better randomization across epochs.
            offset = (step * BATCH_SIZE) % (num_training - BATCH_SIZE)
            batch_data = Xtr[offset:(offset + BATCH_SIZE), :, :, :]
            one_hot_encoding = (np.arange(NUM_LABELS) == Ytr[:, None]).astype(np.float32)
            batch_labels = one_hot_encoding[offset:(offset + BATCH_SIZE)]
            # This dictionary maps the batch data (as a numpy array) to the
            # node in the graph it should be fed to.
            feed_dict = {self.train_data_node: batch_data,
                         self.train_labels_node: batch_labels}
            # Run the graph and fetch some of the nodes.
            _, l, lr, predictions = self.sess.run(
              [self.optimizer, self.loss, self.learning_rate, self.train_prediction],
              feed_dict=feed_dict)

            # Print out the loss periodically.
            if step % 100 == 0:
                error, _ = self.error_rate(predictions, batch_labels)
                print('Step %d of %d' % (step, steps))
                print('Mini-batch loss: %.5f Error: %.5f Learning rate: %.5f' % (l, error, lr))
                print('Validation error: %.1f%%' % self.error_rate(
                      self.validation_prediction.eval(), (np.arange(NUM_LABELS) == Yval[:, None]).astype(np.float32))[0])
