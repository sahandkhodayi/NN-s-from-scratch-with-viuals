class Trainer:

    def __init__(self, network, loss_fn, learning_rate=0.01):
        self.network = network
        self.loss_fn = loss_fn
        self.lr = learning_rate

    def step(self, X: list, Y_true: list):
        """Single forward + backward + weight-update step."""

        # 1. Forward pass
        Y_pred = self.network.predict(X)

        # 2. Compute loss
        loss = self.loss_fn.forward(Y_true, Y_pred)

        # 3. Compute loss gradients (one per output neuron)
        loss_grads = self.loss_fn.backward(Y_true, Y_pred)

        # 4. Backprop through the network
        self.network.backward(loss_grads)

        # 5. Update every weight and bias in every layer
        for layer in self.network.layers:
            for neuron in layer.main_nodes:
                neuron.weight = [
                    w - self.lr * dw
                    for w, dw in zip(neuron.weight, neuron.dW)
                ]
                neuron.bias = neuron.bias - self.lr * neuron.dB

        return loss

    def train_epoch(self, X_data: list, Y_data: list):
        """Run one full pass over the dataset, with shuffling."""
        import random
        pairs = list(zip(X_data, Y_data))
        random.shuffle(pairs)
        total_loss = 0.0
        for X, Y in pairs:
            total_loss += self.step(X, Y)
        return total_loss / len(pairs)

    def train(self, X_data: list, Y_data: list, epochs: int = 100, verbose: bool = True):
        """Train over a dataset for a number of epochs."""
        for epoch in range(1, epochs + 1):
            total_loss = 0.0
            for X, Y in zip(X_data, Y_data):
                total_loss += self.step(X, Y)
            avg_loss = total_loss / len(X_data)
            if verbose and (epoch % max(1, epochs // 10) == 0):
                print(f"Epoch {epoch}/{epochs}  loss={avg_loss:.6f}")
        return avg_loss
