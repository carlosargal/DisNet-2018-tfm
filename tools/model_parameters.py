# ============================ Importing modules ============================ #

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


# =============================== Parameters =============================== #


def custom_params(filters=64, kernel_size=3, strides=2, padding='same', size=None):
    """ Define all the parameters required to create a 2D convolutional layer.
    Args:
        filters: (integer) number of filters used to convolve with the input.
        kernel_size: (integer) size in #pixels wrt one side of the squared filters.
        strides: (integer) displacement in #pixels of the filters per iteration.
        padding: (string) specifies how the input should be padded.
        size: (integer) size of the rescaled image at the output of the 2d deconvolution.
    Return:
        params: (dictionary) contains all the parameters defined previously.
    """
    params = {'filters': filters,
              'kernel_size': kernel_size,
              'strides': strides,
              'padding': padding}

    if size is not None:
        params.update({'size': size})

    return params


class FLG(object):
    # ------------------- Dataset parameters ------------------- #

    train_length = 10000         # Number of training clips.
    test_length = 1000           # Number of validation clips.
    example_length = 20          # Total number of frames per clip (SequenceExample).
    seq_length = 10              # Number of frames in each clip used to learn.
    height = 64                  # Number of pixels on the vertical axis in each frame.
    width = 64                   # Number of pixels on the horizontal axis in each frame.
    channel = 1                  # Desired number of color channels for the frames that feed the network.
    with_bg = False              # It determines whether the dataset has background or not.
    fps = 7                      # Frames per second shown in gif animations.
    num_class = {'m_label': 2,   # Types of movements the symbols do.
                 'fg_label': 2}  # Types of symbols there are.

    # ------------------ Train/eval parameters ----------------- #

    # Configurable hyperparameters:
    batch_size = 20          # Number of clips computed in one step.
    num_epochs = 100         # The network is trained num_epochs times the hole dataset.
    checkpoint_epochs = 5    # Interval in #epochs to save the next checkpoint.
    summary_epochs = 1       # Interval in #epochs to compute the next summaries.
    decay_epochs = 30        # Interval in #epochs to decrease the learning_rate.
    decay_rate = 0.5         # Learning rate decay factor.
    learning_rate = 0.0001   # Initial learning rate.
    num_checkpoints = 3      # Leave the last num_checkpoints saved.
    num_img_outputs = 10     # Amount of frames examples to show in Tensorboard.
    resume_training = False  # Start training from scratch if its False.
    parallel_calls = 10      # Parallelize the map transformation, using up to #CPU cores.
    dropout_rate = 0.25      # Rate of neurons disabled in the FC layers during training.

    # Conversion to steps:
    steps_per_epoch = train_length / batch_size             # Amount of steps computed in each epoch.
    checkpoint_steps = steps_per_epoch * checkpoint_epochs  # Interval in #steps to save the next checkpoint.
    summary_steps = steps_per_epoch * summary_epochs        # Interval in #steps to compute the next summaries.
    train_steps = steps_per_epoch * num_epochs              # Total #steps required to train the network.
    decay_steps = steps_per_epoch * decay_epochs            # Interval in #steps to decrease the learning_rate.

    # ------------------- Network parameters ------------------- #

    # Parameters for conv2d layers:
    conv1_params = custom_params()
    conv2_params = custom_params(filters=128)
    conv3_params = custom_params(filters=128)
    conv4_params = custom_params(filters=256)
    conv5_params = custom_params(filters=256)
    last_conv_params = custom_params(filters=1024, kernel_size=2, padding='valid')

    # deconv_fn layers based on NN-Resize + conv2d:
    deconv1_params = custom_params(filters=256, size=[2, 2])
    deconv2_params = custom_params(filters=256, size=[4, 4])
    deconv3_params = custom_params(filters=128, size=[8, 8])
    deconv4_params = custom_params(filters=128, size=[16, 16])
    deconv5_params = custom_params(filters=64, size=[32, 32])
    last_deconv_params = custom_params(filters=2, size=[height, width])

    # Parameters for CausalConv1D layer:
    temp_params = {'filters': last_conv_params['filters'], 'kernel_size': seq_length}

    # Parameters for batch_norm layers:
    bn_params = {'decay': 0.9, 'epsilon': 1e-5, 'scale': True}

    # Number of neurons used in the fully connected (dense) layers:
    units = last_conv_params['filters']

    # Number of neurons used in the classifier (dense) layer:
    units_classifier = 512

    def __init__(self):
        pass
