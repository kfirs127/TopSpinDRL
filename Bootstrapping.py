import numpy as np
from Evaluation import createsStates
from BWAS import BWAS
import tensorflow as tf
from TopSpin import TopSpin


class Bootstrapping:

    def __init__(self, n, k=3):
        self.bootstraping_heuristic = None
        self.optimizer = None
        self.n = n
        self.k = k

    def set(self, bootstrapping_heuristic, optimizer):
        self.bootstraping_heuristic = bootstrapping_heuristic
        self.optimizer = optimizer

    def get_h_values(self, states):
        states_as_list = [state.get_state_as_list() for state in states]
        states = np.array(states_as_list, dtype=np.float32)
        states_tensor = tf.convert_to_tensor(states)
        predictions = self.bootstraping_heuristic(states_tensor, training=False)
        return predictions.numpy().flatten()

    def bootstrappingTraining(self):
        T_max = 5000

        for i in range(200):
            print_counter = 1
            does_one_solved = False

            random_states = createsStates(list(np.arange(1, self.n + 1)), self.k)
            T = 500
            inputs = []
            outputs = []

            while not does_one_solved:
                for random_state in random_states:
                    if (print_counter - 1) % 50 == 0:
                        print(f"Level {i + 1}, Processing state {print_counter - 1} / {len(random_states)}")
                    print_counter += 1

                    topspin = TopSpin(self.n, self.k, random_state)
                    path, _ = BWAS(topspin, 5, 10, self.bootstraping_heuristic, T)
                    if path is not None:
                        does_one_solved = True
                        counter = len(path) - 1
                        for state in path:
                            inputs.append(state)
                            outputs.append(counter)
                            counter -= 1

                if not does_one_solved:
                    T = max(T * 2, T_max)

            print(f'at level {i + 1} we solved {len(inputs)} times')
            self.train_model(inputs, outputs, epochs=5)

        return self.bootstraping_heuristic


    def train_model(self, input_data, output_labels, epochs=100):
        inputs = np.array(input_data, dtype=np.float32)
        outputs = np.array(output_labels, dtype=np.float32)

        inputs_tensor = tf.convert_to_tensor(inputs)
        outputs_tensor = tf.convert_to_tensor(outputs)

        loss_fn = tf.keras.losses.MeanSquaredError()

        for epoch in range(epochs):
            with tf.GradientTape() as tape:
                predictions = self.bootstraping_heuristic(inputs_tensor, training=True)
                loss = loss_fn(outputs_tensor, predictions)

            gradients = tape.gradient(loss, self.bootstraping_heuristic.trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, self.bootstraping_heuristic.trainable_variables))
            print(f"Epoch {epoch + 1}, Loss: {loss.numpy()}")
