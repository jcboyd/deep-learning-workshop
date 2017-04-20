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

        # Create a new interactive session that we'll use in
        # subsequent code cells.
        self.sess = tf.InteractiveSession()

        # Use our newly created session as the default for 
        # subsequent operations.
        self.sess.as_default()

        self.init_vars()
        self.init_model_architecture()
        self.init_training_graph()

    def init_vars(self):
        # This is where training samples and labels are fed to the graph.
        # These placeholder nodes will be fed a batch of training data at each
        # training step, which we'll write once we define the graph structure.
        self.input_node = tf.placeholder(
            tf.float32,
            shape=(None, self.IMAGE_SIZE, self.IMAGE_SIZE, self.NUM_CHANNELS))

        self.train_labels_node = tf.placeholder(tf.float32,
            shape=(self.BATCH_SIZE, self.NUM_LABELS))

        # The variables below hold all the trainable weights. For each, the
        # parameter defines how the variables will be initialized.
        self.conv1_weights = tf.Variable(
        tf.truncated_normal([5, 5, self.NUM_CHANNELS, 32],  # 5x5 filter, depth 32.
                              stddev=0.1,
                              seed=self.SEED))

        self.conv1_biases = tf.Variable(tf.zeros([32]))

        self.conv2_weights = tf.Variable(
            tf.truncated_normal([5, 5, 32, 64],
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
        """The Model definition."""
        # 2D convolution, with 'SAME' padding (i.e. the output feature map has
        # the same size as the input). Note that {strides} is a 4D array whose
        # shape matches the data layout: [image index, y, x, depth].
        self.conv1 = tf.nn.conv2d(self.input_node, self.conv1_weights,
            strides=[1, 1, 1, 1], padding='SAME')

        # Bias and rectified linear non-linearity.
        self.relu1 = tf.nn.relu(tf.nn.bias_add(self.conv1, self.conv1_biases))

        # Max pooling. The kernel size spec ksize also follows the layout of
        # the data. Here we have a pooling window of 2, and a stride of 2.
        self.pool1 = tf.nn.max_pool(self.relu1,
                              ksize=[1, 2, 2, 1],
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

        # Reshape the feature map cuboid into a 2D matrix to feed it to the
        # fully connected layers.
        # pool_shape = self.pool2.get_shape().as_list()
        pool_shape = tf.shape(self.pool2)
        self.reshape = tf.reshape(self.pool2,
            [pool_shape[0], pool_shape[1] * pool_shape[2] * pool_shape[3]])

        # Fully connected layer. Note that the '+' operation automatically
        # broadcasts the biases.
        self.hidden1 = tf.nn.relu(
            tf.matmul(self.reshape, self.fc1_weights) + self.fc1_biases)

        print('Model architecture initialised')

    def init_training_graph(self):
        # Add a 50% dropout during training only. Dropout also scales
        # activations such that no rescaling is needed at evaluation time.
        dropout = tf.nn.dropout(self.hidden1, self.keep_prob, seed=self.SEED)
        logits = tf.matmul(dropout, self.fc2_weights) + self.fc2_biases

        # Training computation: logits + cross-entropy loss.
        # self.logits = self.model_architecture(self.train_data_node, True)
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
          logits=logits, labels=self.train_labels_node))

        # L2 regularization for the fully connected parameters.
        self.regularizers = (
            tf.nn.l2_loss(self.fc1_weights) + tf.nn.l2_loss(self.fc1_biases) +
            tf.nn.l2_loss(self.fc2_weights) + tf.nn.l2_loss(self.fc2_biases))
        # Add the regularization term to the loss.
        self.loss += 5e-4 * self.regularizers

        # Optimizer: set up a variable that's incremented once per batch and
        # controls the learning rate decay.
        self.batch = tf.Variable(0)

        # Predictions for the minibatch, validation set and test set.
        self.train_prediction = tf.nn.softmax(logits)
        # We'll compute them only once in a while by calling their {eval()} method.
        predictions = tf.matmul(self.hidden1, self.fc2_weights) + self.fc2_biases
        self.validation_prediction = tf.nn.softmax(predictions)
        self.test_prediction = tf.nn.softmax(predictions)

        # Initialize all the variables we defined above.
        tf.global_variables_initializer().run()

        print('Computational graph initialised')


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
        """Return the error rate and confusions."""
        correct = np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
        total = predictions.shape[0]

        error = 100.0 - (100 * float(correct) / float(total))

        confusions = np.zeros([10, 10], np.float32)
        bundled = zip(np.argmax(predictions, 1), np.argmax(labels, 1))
        for predicted, actual in bundled:
            confusions[predicted, actual] += 1

        return error, confusions

    def train(self, Xtr, Ytr, Xval, Yval):
        self.NUM_LABELS = 10

        num_training = Xtr.shape[0]

        # Decay once per epoch, using an exponential schedule starting at 0.01.
        self.learning_rate = tf.train.exponential_decay(
          0.01,                # Base learning rate.
          self.batch * self.BATCH_SIZE,  # Current index into the dataset.
          num_training,        # Decay step.
          0.95,                # Decay rate.
          staircase=True)
        # Use simple momentum for the optimization.
        self.optimizer = tf.train.MomentumOptimizer(self.learning_rate, 0.9).minimize(
            self.loss, global_step=self.batch)
        tf.global_variables_initializer().run()

        # Train over the first 1/4th of our training set.
        steps = num_training // self.BATCH_SIZE
        for step in range(steps):
            # Compute the offset of the current minibatch in the data.
            # Note that we could use better randomization across epochs.
            offset = (step * self.BATCH_SIZE) % (num_training - self.BATCH_SIZE)
            batch_data = Xtr[offset:(offset + self.BATCH_SIZE), :, :, :]
            one_hot_encoding = (
                np.arange(self.NUM_LABELS) == Ytr[:, None]).astype(np.float32)
            batch_labels = one_hot_encoding[offset:(offset + self.BATCH_SIZE)]
            # This dictionary maps the batch data (as a numpy array) to the
            # node in the graph it should be fed to.
            feed_dict = {self.input_node: batch_data,
                         self.train_labels_node: batch_labels}
            # Run the graph and fetch some of the nodes.
            _, l, lr, predictions = self.sess.run(
              [self.optimizer, self.loss, self.learning_rate, self.train_prediction],
              feed_dict=feed_dict)

            # Print out the loss periodically.
            if step % 100 == 0:
                
                error, _ = self.error_rate(predictions, batch_labels)
                print('Step %d of %d' % (step, steps))
                print('Mini-batch loss: %.5f Error: %.5f Learning rate: %.5f' % 
                    (l, error, lr))
                print(
                    'Validation error: %.1f%%' % self.error_rate(
                        self.validation_prediction.eval(
                            feed_dict={self.input_node : Xval}), (
                            np.arange(self.NUM_LABELS) == Yval[:, None]).astype(np.float32))[0])
