import numpy as np
import utils
import typing
import pdb
np.random.seed(1)


def pre_process_images(X: np.ndarray, mean=33.34, stddev=78.59):
    X = normalize(X, mean, stddev)
    X = np.concatenate((X, np.ones((X.shape[0], 1))), axis=1)
    return X


def mean(x):
    return np.mean(x)


def stddev(x):
    return np.std(x)


def normalize(x, mean=33.34, stddev=78.59):
    return (x - mean) / stddev


def sigmoid(z):
    return 1 / (1.0 + np.exp(-z))


def sigmoid_prime(z):
    return sigmoid(z) * (1 - sigmoid(z))


def tanh(z):
    # suggested version of tanh as suggested by LeCun et al.
    return 1.7159 * np.tanh(2 * z / 3)
    # return 1.7159 * np.tanh(2 * z / 3) + 0.01 * z # additional linear term


def tanh_prime(z):
    return 2.28787 / (np.cosh(4 * z / 3) + 1)
    # return 2.28787 / (np.cosh(4 * z / 3) + 1) + 0.01 # additional linear term


def one_hot_encode(Y: np.ndarray, num_classes: int):

    # Borrowed from @D.Samchuk: https://stackoverflow.com/a/49217762
    return np.eye(num_classes, dtype=int)[Y.reshape(-1)]


def cross_entropy_loss(targets: np.ndarray, outputs: np.ndarray) -> float:
    N, K = targets.shape
    cross_error = targets * np.log(outputs)
    assert targets.shape == outputs.shape,\
        f"Targets shape: {targets.shape}, outputs: {outputs.shape}"
    return - (1 / N) * np.sum(cross_error)


class SoftmaxModel:

    def __init__(self,
                 # Number of neurons per layer
                 neurons_per_layer: typing.List[int],
                 use_improved_sigmoid: bool,  # Task 3a hyperparameter
                 use_improved_weight_init: bool  # Task 3c hyperparameter
                 ):
        # Define number of input nodes
        self.I = 785
        self.use_improved_sigmoid = use_improved_sigmoid
        self.use_improved_weight_init = use_improved_weight_init
        # Define number of output nodes
        # neurons_per_layer = [64, 10] indicates that we will have two layers:
        # A hidden layer with 64 neurons and a output layer with 10 neurons.
        self.neurons_per_layer = neurons_per_layer
        #Contains  values of z of the hidden layer
        self.zs= []
        #Same as before but the activation
        self.activations=[]
        # Initialize the weights
        self.ws = []
        prev = self.I
        for size in self.neurons_per_layer:
            w_shape = (prev, size)
            print("Initializing weight to shape:", w_shape)
            if self.use_improved_weight_init:
                sqrt_fan = np.sqrt(w_shape[0])
                w = np.random.uniform(-sqrt_fan, sqrt_fan, w_shape)
            else:
                w = np.random.uniform(-1, 1, w_shape)
            self.ws.append(w)
            prev = size
        self.grads = [None for i in range(len(self.ws))]

    def forward(self, X: np.ndarray) -> np.ndarray:
        # the first propagation is always with the sigmoid function
        z = np.dot(X, self.ws[0])
        self.zs.append(z)
        if self.use_improved_sigmoid:
            activation = tanh(z)
        else:
            activation = sigmoid(z)
        self.activations.append(activation)
        a_k = self.softmax(np.dot(activation, self.ws[1]))
        return a_k

    def softmax(self, z):
        return np.exp(z) / np.sum(np.exp(z), axis=1, keepdims=True)

    def backward(self, X: np.ndarray, outputs: np.ndarray,
                 targets: np.ndarray) -> None:
        assert targets.shape == outputs.shape,\
            f"Output shape: {outputs.shape}, targets: {targets.shape}"

        N, K = targets.shape
        cost_derivate = - (targets - outputs) / N # the error for the output layer
        delta = np.dot(self.activations[-1].T, cost_derivate) # error multiplied by activation of previous layer
        if self.use_improved_sigmoid:
            cost_hiddenL = np.dot(cost_derivate, self.ws[1].T) * tanh_prime(self.zs[-1])
        else:
            cost_hiddenL = np.dot(cost_derivate, self.ws[1].T) * sigmoid_prime(self.zs[-1])
        delta_hiddenL = np.dot(X.T, cost_hiddenL)
        # A list of gradients.
        # For example, self.grads[0] will be the gradient for the first hidden layer
        self.grads = []
        self.grads.append(delta_hiddenL)
        self.grads.append(delta)

        for grad, w in zip(self.grads, self.ws):
            assert grad.shape == w.shape,\
                f"Expected the same shape. Grad shape: {grad.shape}, w: {w.shape}."

    def zero_grad(self) -> None:
        self.grads = [None for i in range(len(self.ws))]


def gradient_approximation_test(
        model: SoftmaxModel, X: np.ndarray, Y: np.ndarray):
    """
        Numerical approximation for gradients. Should not be edited.
        Details about this test is given in the appendix in the assignment.
    """
    epsilon = 1e-3
    for layer_idx, w in enumerate(model.ws):
        for i in range(w.shape[0]):
            for j in range(w.shape[1]):
                orig = model.ws[layer_idx][i, j].copy()
                model.ws[layer_idx][i, j] = orig + epsilon
                logits = model.forward(X)
                cost1 = cross_entropy_loss(Y, logits)
                model.ws[layer_idx][i, j] = orig - epsilon
                logits = model.forward(X)
                cost2 = cross_entropy_loss(Y, logits)
                gradient_approximation = (cost1 - cost2) / (2 * epsilon)
                model.ws[layer_idx][i, j] = orig
                # Actual gradient
                logits = model.forward(X)
                model.backward(X, logits, Y)
                difference = gradient_approximation - \
                    model.grads[layer_idx][i, j]
                assert abs(difference) <= epsilon**2,\
                    f"Calculated gradient is incorrect. " \
                    f"Layer IDX = {layer_idx}, i={i}, j={j}.\n" \
                    f"Approximation: {gradient_approximation}, actual gradient: {model.grads[layer_idx][i, j]}\n" \
                    f"If this test fails there could be errors in your cross entropy loss function, " \
                    f"forward function or backward function"


if __name__ == "__main__":
    # Simple test on one-hot encoding
    Y = np.zeros((1, 1), dtype=int)
    Y[0, 0] = 3
    Y = one_hot_encode(Y, 10)
    assert Y[0, 3] == 1 and Y.sum() == 1, \
        f"Expected the vector to be [0,0,0,1,0,0,0,0,0,0], but got {Y}"

    X_train, Y_train, *_ = utils.load_full_mnist(0.1)
    mean = mean(X_train)
    stddev = stddev(X_train)
    X_train = pre_process_images(X_train, mean, stddev)
    Y_train = one_hot_encode(Y_train, 10)
    assert X_train.shape[1] == 785,\
        f"Expected X_train to have 785 elements per image. Shape was: {X_train.shape}"

    neurons_per_layer = [64, 10]
    use_improved_sigmoid = False
    use_improved_weight_init = False
    model = SoftmaxModel(
        neurons_per_layer, use_improved_sigmoid, use_improved_weight_init)

    # Gradient approximation check for 100 images
    X_train = X_train[:100]
    Y_train = Y_train[:100]
    for layer_idx, w in enumerate(model.ws):
        model.ws[layer_idx] = np.random.uniform(-1, 1, size=w.shape)

    gradient_approximation_test(model, X_train, Y_train)
