import numpy as np


class LinearClassifier:

    def __init__(self, X, y):
        self.Xtr = X
        self.Ytr = y
        
        self.N, self.D = self.Xtr.shape
        self.K = max(y) - min(y) + 1

        np.random.seed(0)
        self.W = np.random.rand(self.D, self.K) * 0.001  # random weight vector

    def loss(X, y, W, reg):
        pass

    def gradient():
        pass

    def train(self, reg, batch_size, learning_rate, max_iters, Xval, Yval):
        for iteration in range(max_iters):
            idx = np.random.choice(self.N, batch_size, replace=True)
            X_batch = self.Xtr[idx,:]
            y_batch = self.Ytr[idx]

            dW = self.gradient(self.W, X_batch, y_batch, reg)        
            self.W -= learning_rate * dW

            if iteration % 100 == 0:
                print 'Step %s of %s' % (iteration, max_iters)
                loss = self.loss(X_batch, y_batch, self.W, reg)

                print 'Mini-batch loss: %.05f ' % loss + \
                    'Learning rate: %.05f' % learning_rate

                predictions = [self.predict(Xval[i]) for i in range(len(Yval))]
                print 'Validation error: %.04f' % (
                    float(sum(Yval != predictions)) / len(Yval))

    def train_iteration(self, reg, batch_size, learning_rate):
        idx = np.random.choice(self.N, batch_size, replace=True)
        X_batch = self.Xtr[idx,:]
        y_batch = self.Ytr[idx]
    
        dW = self.gradient(self.W, X_batch, y_batch, reg)       
        self.W -= learning_rate * dW
        return self.loss(X_batch, y_batch, self.W, reg)

    def predict(self, Xte):
        return np.argmax(np.dot(Xte, self.W))


class MultiSVM(LinearClassifier):

    def loss(self, X, y, W, reg):
        """
        fully-vectorised implementation:
        - X holds all the training as rows (e.g. 50,000 x 3073 in 
          CIFAR-10)
        - y is an array of integers specifying correct class (e.g. 
          50,000 x 1 array)
        - W are weights (e.g. 3073 x 10)
        """
        delta = 1.0
        N, D = X.shape
        scores =  X.dot(W)
        # compute the margins for all classes in one vector operation
        margins = np.maximum(0, scores.T - scores[np.arange(N), y] + delta)
        margins[y, np.arange(N)] -= delta
        # Sum over all margins and adjust for extraneous deltas
        loss = np.sum(margins)
        return loss / N + 0.5 * reg * np.sum(W * W)

    def gradient(self, W, X, y, reg):
        """
        fully-vectorised implementation:
        - X holds all the training as rows (e.g. 50,000 x 3073 in 
          CIFAR-10)
        - y is an array of integers specifying correct class (e.g. 
          50,000 x 1 array)
        - W are weights (e.g. 3073 x 10)
        """
        delta = 1.0
        N, D = X.shape
        scores = X.dot(W)  # (50,000 x 3,000) x (3,000 x 10) => (50,000 x 10)
        margins = np.maximum(0, (scores.T - scores[np.arange(N), y] + delta)).T
        margins[np.arange(N), y] -= delta

        binaries = margins
        binaries[margins > 0] = 1

        binaries[np.arange(N), y] =- np.sum(binaries, axis=1)
        return np.dot(X.T, binaries) / N + reg * W


class SoftmaxRegression(LinearClassifier):

    def loss(self, X, y, W, reg):
        N, D = X.shape

        scores = np.dot(X, W)
        exp_scores = np.exp(scores)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

        correct_logprobs = -np.log(probs[np.arange(N), y])  # advanced slicing

        data_loss = np.sum(correct_logprobs) / N
        reg_loss = 0.5 * reg * np.sum(W * W)
        loss = data_loss + reg_loss
        return loss

    def gradient(self, W, X, y, reg):
        """
        For $L_i = -\log(p_{y_i})$, $$\frac{\partial L_i}{\partial f_k} = -\frac{1}{p_{y_i}}\cdot\frac{\partial p_{y_i}}{\partial f_k}$$ 
        For $k = y_{i}$, $$\frac{\partial p_{y_i}}{\partial f_k} = \frac{\partial}{\partial f_k}\frac{e^{f_{y_i}}}{\sum_j e^{f_j}} = p_{y_i} - p_{y_i}^2 \implies \frac{\partial L_i}{\partial f_k} = p_{y_i} - 1$$
        For $k \neq y_{i}$, $$\frac{\partial p_{y_i}}{\partial f_k} = -p_kp_{y_i} \implies \frac{\partial L_i}{\partial f_k} = p_{y_i}$$
        Hence, in general, $$\frac{\partial L_i}{\partial f_k} = p_{y_i} - 1(k = y_i)$$
        """
        N, D = X.shape

        scores = np.dot(X, W)
        exp_scores = np.exp(scores)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

        dscores = probs
        dscores[np.arange(N), y] -= 1
        dscores /= N

        dW = np.dot(X.T, dscores) + reg * W
        return dW
