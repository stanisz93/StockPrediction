import tensorflow as tf
from tensorflow.keras import utils, layers
from tensorflow_probability import distributions as tfd
from constants.configs import PO_PARAMETERS, COMPONENTS


def nnelu(input):
    """ Computes the Non-Negative Exponential Linear Unit
    """
    return tf.add(tf.constant(1, dtype=tf.float32), tf.nn.elu(input))


def slice_parameter_vectors(parameter_vector):
    """ Returns an unpacked list of paramter vectors.
    """
    return [parameter_vector[:, i*COMPONENTS:(i+1)*COMPONENTS] for i in range(PO_PARAMETERS)]


def gnll_loss(y, parameter_vector):
    """ Computes the mean negative log-likelihood loss of y given the mixture parameters.
    """
    alpha, mu, sigma = slice_parameter_vectors(
        parameter_vector)  # Unpack parameter vectors

    gm = tfd.MixtureSameFamily(
        mixture_distribution=tfd.Categorical(probs=alpha),
        components_distribution=tfd.Normal(
            loc=mu,
            scale=sigma))

    # Evaluate log-probability of y
    log_likelihood = gm.log_prob(tf.transpose(y))

    return -tf.reduce_mean(log_likelihood, axis=-1)

utils.get_custom_objects().update({'nnelu': layers.Activation(nnelu)})
