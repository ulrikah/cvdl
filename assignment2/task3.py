import numpy as np
import utils
import time
import matplotlib.pyplot as plt
import typing
from task2a import cross_entropy_loss, SoftmaxModel, one_hot_encode, pre_process_images, mean, stddev
np.random.seed(0)


def calculate_accuracy(X: np.ndarray, targets: np.ndarray,
                       model: SoftmaxModel) -> float:
    """
    Args:
        X: images of shape [batch size, 785]
        targets: labels/targets of each image of shape: [batch size, 10]
        model: model of class SoftmaxModel
    Returns:
        Accuracy (float)
    """
    outputs = model.forward(X)
    return np.mean(np.argmax(targets, axis=1) == np.argmax(outputs, axis=1))


def shuffle(X: np.ndarray, Y: np.ndarray):
    concat = np.concatenate((X_train, Y_train), axis=1)
    np.random.shuffle(concat)
    return concat[:, 0:X.shape[1]], concat[:,X.shape[1]:]


def train(
        model: SoftmaxModel,
        datasets: typing.List[np.ndarray],
        num_epochs: int,
        learning_rate: float,
        batch_size: int,
        # Task 3 hyperparameters,
        use_shuffle: bool,
        use_momentum: bool,
        momentum_gamma: float):
    X_train, Y_train, X_val, Y_val, X_test, Y_test = datasets

    # Utility variables
    num_batches_per_epoch = X_train.shape[0] // batch_size
    num_steps_per_val = num_batches_per_epoch // 5
    # Tracking variables to track loss / accuracy
    train_loss = {}
    val_loss = {}
    train_accuracy = {}
    val_accuracy = {}

    global_step = 0
    for epoch in range(num_epochs):
        print("Epoch", epoch)
        if use_shuffle:
            X_train, Y_train = shuffle(X_train, Y_train)
        for step in range(num_batches_per_epoch):
            start = step * batch_size
            end = start + batch_size
            X_batch, Y_batch = X_train[start:end], Y_train[start:end]

            # Track train / validation loss / accuracy
            # every time we progress 20% through the dataset
            outputs = model.forward(X_batch)
            model.backward(X_batch, outputs, Y_batch)

            # update weigths
            if use_momentum:
                model.ws[-1] = momentum_gamma * model.ws[-1] - learning_rate * model.grads[-1]
                model.ws[-2] = momentum_gamma * model.ws[-2] - learning_rate * model.grads[-2]
            else:
                model.ws[-1] = model.ws[-1] - learning_rate * model.grads[-1]
                model.ws[-2] = model.ws[-2] - learning_rate * model.grads[-2]

            if (global_step % num_steps_per_val) == 0:
                _outputs_train = model.forward(X_train)
                _train_loss = cross_entropy_loss(Y_train, _outputs_train)
                train_loss[global_step] = _train_loss

                _outputs_val = model.forward(X_val)
                _val_loss = cross_entropy_loss(Y_val, _outputs_val)
                val_loss[global_step] = _val_loss

                train_accuracy[global_step] = calculate_accuracy(
                    X_train, Y_train, model)
                val_accuracy[global_step] = calculate_accuracy(
                    X_val, Y_val, model)

            global_step += 1
    return model, train_loss, val_loss, train_accuracy, val_accuracy


if __name__ == "__main__":
    # Load dataset
    validation_percentage = 0.2
    X_train, Y_train, X_val, Y_val, X_test, Y_test = utils.load_full_mnist(
        validation_percentage)

    # Hyperparameters
    num_epochs = 20
    learning_rate = .1
    batch_size = 32
    neurons_per_layer = [64, 10]
    # neurons_per_layer = [16, 10]
    # neurons_per_layer = [128, 10]
    momentum_gamma = .9  # Task 3 hyperparameter

    # Settings for task 3. Keep all to false for task 2.
    use_shuffle = True
    use_improved_sigmoid = True
    use_improved_weight_init = True
    use_momentum = False

    # advice from the assignment text
    if use_momentum:
        learning_rate = 0.02
        print("Using momentum")
        print("Momentum:", momentum_gamma)
        print("learning_rate:", learning_rate)

    # Calibration
    m = mean(X_train)
    std = stddev(X_train)
    X_train = pre_process_images(X_train, m, std)
    X_val = pre_process_images(X_val, m, std)
    X_test = pre_process_images(X_test, m, std)
    Y_train = one_hot_encode(Y_train, 10)
    Y_val = one_hot_encode(Y_val, 10)
    Y_test = one_hot_encode(Y_test, 10)

    model = SoftmaxModel(
        neurons_per_layer,
        use_improved_sigmoid,
        use_improved_weight_init)

    start = time.time()

    model, train_loss, val_loss, train_accuracy, val_accuracy = train(
        model,
        [X_train, Y_train, X_val, Y_val, X_test, Y_test],
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        batch_size=batch_size,
        use_shuffle=use_shuffle,
        use_momentum=use_momentum,
        momentum_gamma=momentum_gamma)

    m, s = divmod(time.time() - start, 60)
    print("Training took", m, "minutes and", s, "seconds")

    print("Final Train Cross Entropy Loss:",
          cross_entropy_loss(Y_train, model.forward(X_train)))
    print("Final Validation Cross Entropy Loss:",
          cross_entropy_loss(Y_val, model.forward(X_val)))
    print("Final Test Cross Entropy Loss:",
          cross_entropy_loss(Y_test, model.forward(X_test)))

    print("Final Train accuracy:",
          calculate_accuracy(X_train, Y_train, model))
    print("Final Validation accuracy:",
          calculate_accuracy(X_val, Y_val, model))
    print("Final Test accuracy:",
          calculate_accuracy(X_test, Y_test, model))

    # Plot loss
    plt.figure(figsize=(20, 8))
    plt.subplot(1, 2, 1)
    plt.ylim([0.1, .5])
    utils.plot_loss(train_loss, "Training Loss")
    utils.plot_loss(val_loss, "Validation Loss")
    plt.xlabel("Number of gradient steps")
    plt.ylabel("Cross Entropy Loss")
    plt.legend()
    plt.subplot(1, 2, 2)

    # Plot accuracy
    plt.ylim([0.9, 1.0])
    utils.plot_loss(train_accuracy, "Training Accuracy")
    utils.plot_loss(val_accuracy, "Validation Accuracy")
    plt.legend()
    plt.xlabel("Number of gradient steps")
    plt.ylabel("Accuracy")
    plt.savefig("softmax_train_graph.png")
    plt.show()
