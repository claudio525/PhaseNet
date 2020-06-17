import logging
from typing import Dict

import numpy as np

import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()

from .model import Model
from .data_reader import Config

DEFAULT_CONFIG_DICT = {
    "X_shape": (3000, 1, 3),
    "Y_shape": (3000, 1, 3),
    "depth": 5,
    "filters_root": 8,
    "kernel_size": [7, 1],
    "pool_size": [4, 1],
    "dilation_rate": [1, 1],
    "batch_size": 20,
    "class_weights": [1, 1, 1],
    "loss_type": "cross_entropy",
    "weight_decay": 0,
    "optimizer": "adam",
    "learning_rate": 0.01,
    "decay_step": -1,
    "decay_rate": 0.9,
    "momentum": 0.9,
    "summary": True,
    "drop_rate": 0,
}


def predict(input_data: np.ndarray, model_dir: str, config_dict=None):
    """Functions that runs p & s wave estimation
    using the PhaseNet model

    Parameters
    ----------
    input_data: numpy array of floats
        The acceleration series data, supports multiple
        reocrds (first axis), each with the same number
        of timesteps and 3 channels (2 horizontal + vertical)
        Shape: [n_records, n_timesteps, 3]
    model_dir: str
        Path the model directory, i.e. /model/190703-214543
    config_dict: dictionary, optional
        config dictionary, mainly useful for
        adjusting acceleration series length,
        i.e. updating X_shape

    Returns
    -------
    numpy array of floats:
        The noise, p & s wave probabilities at each timestep
        Shape: [n_records, n_timesteps, 3]
    """
    assert len(input_data.shape) == 3

    if config_dict is None:
        config_dict = DEFAULT_CONFIG_DICT

    input_data = input_data[:, :, np.newaxis, :]
    config_dict["X_shape"] = config_dict["Y_shape"] = input_data.shape[1:]
    # input_data = np.transpose(input_data, axes=(1, 0, 2))

    # Normalize each channel of each record
    input_data = (input_data - np.mean(input_data, axis=1, keepdims=True)) / np.std(input_data, axis=1, keepdims=True)

    # Adjust missing channels ??
    channel_max = np.max(np.abs(input_data), axis=1, keepdims=True)
    input_data *= input_data.shape[-1] / np.count_nonzero(channel_max, axis=-1)[..., np.newaxis]

    config = Config()
    config.set_config(config_dict)

    model = Model(config, mode="pred")
    sess_config = tf.ConfigProto()
    sess_config.gpu_options.allow_growth = True
    sess_config.log_device_placement = False

    with tf.Session(config=sess_config) as sess:
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=5)
        init = tf.global_variables_initializer()
        sess.run(init)

        logging.info("Restoring models...")
        latest_check_point = tf.train.latest_checkpoint(model_dir)
        saver.restore(sess, latest_check_point)

        pred_batch = sess.run(
            model.preds,
            feed_dict={
                model.X: input_data,
                model.drop_rate: 0,
                model.is_training: False,
            },
        )

    return pred_batch[:, :, 0, :]
