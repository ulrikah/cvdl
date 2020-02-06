import numpy as np
import utils
import typing
import pdb
np.random.seed(1)


def pre_process_images(X: np.ndarray, mean, stddev):

    X = normalize(X, mean, stddev)
    X = np.concatenate((X, np.ones((X.shape[0], 1))), axis=1)
    return X


def mean(x):
    return np.mean(x)


def stddev(x):
    return np.std(x)


def normalize(x, mean, stddev):
    return (x - mean) / stddev


def sigmoid(z): 
    return 1/(1.0+np.exp(-z))


def sigmoid_prime(z):
	return sigmoid(z)*(1-sigmoid(z))

def one_hot_encode(Y: np.ndarray, num_classes: int):

    # Borrowed from @D.Samchuk: https://stackoverflow.com/a/49217762
    return np.eye(num_classes, dtype=int)[Y.reshape(-1)]


def cross_entropy_loss(targets: np.ndarray, outputs: np.ndarray) -> float:

    N,K=targets.shape
    cross_error= targets*np.log(outputs)
    assert targets.shape == outputs.shape,\
        f"Targets shape: {targets.shape}, outputs: {outputs.shape}"
    return -np.mean(cross_error)


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
            w = np.zeros(w_shape)
            self.ws.append(w)
            prev = size
        self.grads = [None for i in range(len(self.ws))]

    def forward(self, X: np.ndarray) -> np.ndarray:
    	#the first propagation is always with the sigmoid function
    	z= np.dot(X,self.ws[0])
    	self.zs.append(z)
    	activation= sigmoid(z)
    	self.activations.append(activation)
    	a_k=self.softmax(np.dot(activation,self.ws[1]))
    	return a_k

    def softmax(self, z):
        return np.exp(z) / np.sum(np.exp(z), axis=1, keepdims=True)

    def backward(self, X: np.ndarray, outputs: np.ndarray,
                 targets: np.ndarray) -> None:
        """
        Args:
            X: images of shape [batch size, 785]
            outputs: outputs of model of shape: [batch size, num_outputs]
            targets: labels/targets of each image of shape: [batch size, num_classes]
        """
        assert targets.shape == outputs.shape,\
            f"Output shape: {outputs.shape}, targets: {targets.shape}"

        N = targets.shape[0]
        K = targets.shape[1]
        cost_derivate = -(targets - outputs) #the error for the output layer
        delta=np.dot(self.activations[-1].transpose(),cost_derivate) #error multiplied by activation of previous layer
        cost_hiddenL=np.dot(cost_derivate,self.ws[1].transpose())*sigmoid_prime(self.zs[-1])
        delta_hiddenL=np.dot(X.transpose(),cost_hiddenL)
        # A list of gradients.
        # For example, self.grads[0] will be the gradient for the first hidden layer
        self.grads = []
        self.grads.append(delta_hiddenL/(N*K))
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
    logits = model.forward(X_train)
    np.testing.assert_almost_equal(
        logits.mean(), 1/10,
        err_msg="Since the weights are all 0's, the softmax activation should be 1/10")

    # Gradient approximation check for 100 images
    X_train = X_train[:100]
    Y_train = Y_train[:100]
    for i in range(2):
        if i != 0:
            gradient_approximation_test(model, X_train, Y_train)
        model.ws = [np.random.randn(*w.shape) for w in model.ws]
#        model.w = np.random.randn(*model.w.shape)
