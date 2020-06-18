import os
import logging
from typing import Dict
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()

import tensorflow.python.util.deprecation as deprecation

deprecation._PRINT_DEPRECATION_WARNINGS = False

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

DEFAULT_MODEL_DIR = str(Path(__file__).parent / "../model/190703-214543")


def predict(input_data: np.ndarray, model_dir: str = None, config_dict: Dict = None):
    """Functions that runs p & s wave estimation
    using the PhaseNet model

    Parameters
    ----------
    input_data: numpy array of floats
        The acceleration series data, supports multiple
        records (first axis), each with the same number
        of timesteps and 3 channels (2 horizontal + vertical)
        Shape: [n_records, n_timesteps, 3]

        Note: The input data is expected to be at a sample
        rate of 100Hz
    model_dir: str, optional
        Path the model directory,
        Defaults to the included model
    config_dict: dictionary, optional
        model config dictionary, shouldn't generally
        need to change the default unless the
        non-default model is used

    Returns
    -------
    numpy array of floats:
        The noise, p & s wave probabilities at each timestep
        Shape: [n_records, n_timesteps, 3]
    """
    assert len(input_data.shape) == 3

    config_dict = config_dict if config_dict is not None else DEFAULT_CONFIG_DICT
    model_dir = model_dir if model_dir is not None else DEFAULT_MODEL_DIR

    input_data = input_data[:, :, np.newaxis, :]
    config_dict["X_shape"] = config_dict["Y_shape"] = input_data.shape[1:]

    # Normalize each channel of each record
    input_data = (input_data - np.mean(input_data, axis=1, keepdims=True)) / np.std(
        input_data, axis=1, keepdims=True
    )

    # Adjust missing channels ??
    channel_max = np.max(np.abs(input_data), axis=1, keepdims=True)
    input_data *= (
        input_data.shape[-1] / np.count_nonzero(channel_max, axis=-1)[..., np.newaxis]
    )

    config = Config()
    config.set_config(config_dict)

    # Reset
    tf.compat.v1.reset_default_graph()

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


def gen_plot(acc: np.ndarray, t: np.ndarray, probs: np.ndarray, title: str = None):
    """Generates a plot showing the acceleration timeseries,
    the p & s wave pick, and the associated probabilities
    """
    # Create the plot
    fig = plt.figure(figsize=(22, 10))

    # Extract results
    noise_p = probs[0, :, 0]
    p_wave_p = probs[0, :, 1]
    s_wave_p = probs[0, :, 2]
    p_wave_ix, s_wave_ix = np.argmax(probs[0, :, 1]), np.argmax(probs[0, :, 2])

    axes = []
    for ix in range(3):
        ax = (
            fig.add_subplot(4, 1, ix + 1)
            if ix == 0
            else fig.add_subplot(4, 1, ix + 1, sharex=axes[0], sharey=axes[0])
        )
        axes.append(ax)

        ax.plot(t, acc[:, ix])
        ax.axvline(x=t[p_wave_ix], c="orange", linestyle="--", linewidth=1.25)
        ax.axvline(x=t[s_wave_ix], c="green", linestyle="--", linewidth=1.25)

    ax = fig.add_subplot(4, 1, 4, sharex=axes[0])
    ax.plot(t, noise_p, label="noise")
    ax.plot(t, p_wave_p, label=f"p-wave - {p_wave_p[p_wave_ix]:.2f}")
    ax.plot(t, s_wave_p, label=f"s-wave - {s_wave_p[s_wave_ix]:.2f}")

    ax.legend()

    if title is not None:
        axes[0].set_title(title)

    fig.tight_layout()
    plt.subplots_adjust(hspace=0)
    return fig
