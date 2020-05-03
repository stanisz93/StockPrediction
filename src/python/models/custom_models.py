import tensorflow as tf
import numpy as np
from constants.configs import PARAMS
from models.utils import losses


def simple_lstm_model(trainX: np.array):
    return tf.keras.models.Sequential([
        tf.keras.layers.LSTM(PARAMS["hidden_size"],
                             input_shape=trainX.shape[-2:]),
        tf.keras.layers.Dense(1)])


def mixture_model(trainX: np.array):
    components = 2
    input_1 = tf.keras.layers.Input(shape=trainX.shape[-2:])
    lstm = tf.keras.layers.LSTM(PARAMS["hidden_size"])(input_1)
    alphas = tf.keras.layers.Dense(
        components, activation="softmax", name="alphas")(lstm)
    mus = tf.keras.layers.Dense(components, name="mus")(lstm)
    sigmas = tf.keras.layers.Dense(
        components, activation="nnelu", name="sigmas")(lstm)
    mixture_lstm = tf.keras.layers.Concatenate(
        name="output")([alphas, mus, sigmas])
    return tf.keras.models.Model(inputs=[input_1], outputs=[mixture_lstm])


model_loss_dict = {"lstm": (simple_lstm_model, "mae"),
                   "mixture_lstm": (mixture_model, losses.gnll_loss)}
