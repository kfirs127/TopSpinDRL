import tensorflow as tf
from tensorflow.keras import layers, regularizers


class Actor(tf.keras.Model):
    def __init__(self):
        super(Actor, self).__init__()
        self.common0 = layers.Dense(256, kernel_regularizer=regularizers.l2(0.01))
        self.leaky_relu0 = layers.LeakyReLU(alpha=0.01)
        self.dropout0 = layers.Dropout(0.5)

        self.common1 = layers.Dense(128, kernel_regularizer=regularizers.l2(0.01))
        self.leaky_relu1 = layers.LeakyReLU(alpha=0.01)
        self.dropout1 = layers.Dropout(0.5)

        self.common2 = layers.Dense(64, kernel_regularizer=regularizers.l2(0.01))
        self.leaky_relu2 = layers.LeakyReLU(alpha=0.01)
        self.dropout2 = layers.Dropout(0.25)

        self.common3 = layers.Dense(32, kernel_regularizer=regularizers.l2(0.01))
        self.leaky_relu3 = layers.LeakyReLU(alpha=0.01)
        self.dropout3 = layers.Dropout(0.25)

        self.common4 = layers.Dense(16, kernel_regularizer=regularizers.l2(0.01))
        self.leaky_relu4 = layers.LeakyReLU(alpha=0.01)
        self.dropout4 = layers.Dropout(0.1)

        self.actor = layers.Dense(3, activation='softmax')

    def call(self, inputs, training=False):
        x = self.common0(inputs)
        x = self.leaky_relu0(x)
        x = self.dropout0(x, training=training)

        x = self.common1(x)
        x = self.leaky_relu1(x)
        x = self.dropout1(x, training=training)

        x = self.common2(x)
        x = self.leaky_relu2(x)
        x = self.dropout2(x, training=training)

        x = self.common3(x)
        x = self.leaky_relu3(x)
        x = self.dropout3(x, training=training)

        x = self.common4(x)
        x = self.leaky_relu4(x)
        x = self.dropout4(x, training=training)

        return self.actor(x)


class Critic(tf.keras.Model):
    def __init__(self):
        super(Critic, self).__init__()
        self.common0 = layers.Dense(256, kernel_regularizer=regularizers.l2(0.01))
        self.leaky_relu0 = layers.LeakyReLU(alpha=0.01)
        self.dropout0 = layers.Dropout(0.5)

        self.common1 = layers.Dense(128, kernel_regularizer=regularizers.l2(0.01))
        self.leaky_relu1 = layers.LeakyReLU(alpha=0.01)
        self.dropout1 = layers.Dropout(0.5)

        self.common2 = layers.Dense(64, kernel_regularizer=regularizers.l2(0.01))
        self.leaky_relu2 = layers.LeakyReLU(alpha=0.01)
        self.dropout2 = layers.Dropout(0.25)

        self.common3 = layers.Dense(32, kernel_regularizer=regularizers.l2(0.01))
        self.leaky_relu3 = layers.LeakyReLU(alpha=0.01)
        self.dropout3 = layers.Dropout(0.25)

        self.common4 = layers.Dense(16, kernel_regularizer=regularizers.l2(0.01))
        self.leaky_relu4 = layers.LeakyReLU(alpha=0.01)
        self.dropout4 = layers.Dropout(0.1)

        self.critic = layers.Dense(1)

    def call(self, inputs, training=False):
        x = self.common0(inputs)
        x = self.leaky_relu0(x)
        x = self.dropout0(x, training=training)

        x = self.common1(x)
        x = self.leaky_relu1(x)
        x = self.dropout1(x, training=training)

        x = self.common2(x)
        x = self.leaky_relu2(x)
        x = self.dropout2(x, training=training)

        x = self.common3(x)
        x = self.leaky_relu3(x)
        x = self.dropout3(x, training=training)

        x = self.common4(x)
        x = self.leaky_relu4(x)
        x = self.dropout4(x, training=training)

        return self.critic(x)


class DuelingActorCritic(tf.keras.Model):
    def __init__(self):
        super(DuelingActorCritic, self).__init__()
        self.common0 = layers.Dense(256, kernel_regularizer=regularizers.l2(0.01))
        self.leaky_relu0 = layers.LeakyReLU(alpha=0.01)
        self.dropout0 = layers.Dropout(0.5)

        self.common1 = layers.Dense(128, kernel_regularizer=regularizers.l2(0.01))
        self.leaky_relu1 = layers.LeakyReLU(alpha=0.01)
        self.dropout1 = layers.Dropout(0.5)

        self.common2 = layers.Dense(64, kernel_regularizer=regularizers.l2(0.01))
        self.leaky_relu2 = layers.LeakyReLU(alpha=0.01)
        self.dropout2 = layers.Dropout(0.25)

        self.common3 = layers.Dense(32, kernel_regularizer=regularizers.l2(0.01))
        self.leaky_relu3 = layers.LeakyReLU(alpha=0.01)
        self.dropout3 = layers.Dropout(0.25)

        self.common4 = layers.Dense(16, kernel_regularizer=regularizers.l2(0.01))
        self.leaky_relu4 = layers.LeakyReLU(alpha=0.01)
        self.dropout4 = layers.Dropout(0.1)

        self.critic = layers.Dense(1)
        self.actor = layers.Dense(3, activation='softmax')

    def call(self, inputs, training=False):
        x = self.common0(inputs)
        x = self.leaky_relu0(x)
        x = self.dropout0(x, training=training)

        x = self.common1(x)
        x = self.leaky_relu1(x)
        x = self.dropout1(x, training=training)

        x = self.common2(x)
        x = self.leaky_relu2(x)
        x = self.dropout2(x, training=training)

        x = self.common3(x)
        x = self.leaky_relu3(x)
        x = self.dropout3(x, training=training)

        x = self.common4(x)
        x = self.leaky_relu4(x)
        x = self.dropout4(x, training=training)

        return self.actor(x), self.critic(x)

class RewardModel(tf.keras.Model):
    def __init__(self):
        super(RewardModel, self).__init__()
        self.common0 = layers.Dense(256, kernel_regularizer=regularizers.l2(0.01))
        self.leaky_relu0 = layers.LeakyReLU(alpha=0.01)
        self.dropout0 = layers.Dropout(0.5)

        self.common1 = layers.Dense(128, kernel_regularizer=regularizers.l2(0.01))
        self.leaky_relu1 = layers.LeakyReLU(alpha=0.01)
        self.dropout1 = layers.Dropout(0.5)

        self.common2 = layers.Dense(64, kernel_regularizer=regularizers.l2(0.01))
        self.leaky_relu2 = layers.LeakyReLU(alpha=0.01)
        self.dropout2 = layers.Dropout(0.25)

        self.common3 = layers.Dense(32, kernel_regularizer=regularizers.l2(0.01))
        self.leaky_relu3 = layers.LeakyReLU(alpha=0.01)
        self.dropout3 = layers.Dropout(0.25)

        self.reward = layers.Dense(1)

    def call(self, inputs, training=False):
        x = self.common0(inputs)
        x = self.leaky_relu0(x)
        x = self.dropout0(x, training=training)

        x = self.common1(x)
        x = self.leaky_relu1(x)
        x = self.dropout1(x, training=training)

        x = self.common2(x)
        x = self.leaky_relu2(x)
        x = self.dropout2(x, training=training)

        x = self.common3(x)
        x = self.leaky_relu3(x)
        x = self.dropout3(x, training=training)

        return self.reward(x)