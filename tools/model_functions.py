# ============================ Importing modules ============================ #

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import moviepy.editor as mpy
import numpy as np
import os
import tempfile
import tensorflow as tf
import StringIO

from tensorflow.python.framework import constant_op
from tensorflow.python.ops import summary_op_util
from tools.model_parameters import FLG   # Class containing all the required parameters to run the models.

# ============================= Input pipeline ============================= #


def decode_clip(frame_list, h, w, l):
    """ Adapt the frames of each clip to feed the network with the format and value ranges expected.
    Args:
        frame_list: (list) each element is an array with the frame values in range [0, 255].
        h: (integer) number of pixels on the vertical axis in each frame.
        w: (integer) number of pixels on the horizontal axis in each frame.
        l: (integer) number of frames per clip (SequenceExample).
    Return:
        tensor containing the decoded clip within the value range [0, 1].
    """
    clip = []

    for i in range(l):
        frame = frame_list[i]

        # Decode the jpeg frames contained in the TFRecords files.
        image_decoded = tf.image.decode_jpeg(frame, channels=FLG.channel)

        # Map the value ranges from [0, 255] to [0, 1] (rescale) and convert to float32.
        image_converted = tf.image.convert_image_dtype(image_decoded, tf.float32)

        # Ensure all the frames have the same size that the model expects.
        image_resized = tf.image.resize_images(image_converted, [h, w])

        # Append each resulting frame inside clip list.
        clip.append(image_resized)

    return tf.stack(clip)


def generate_mask(img_mask_list, h, w, l):
    """ create a bounding box containing the target object at the same time
        that compute the relative loss of its surface wrt the size of the frame.
    Args:
        img_mask_list: (list) each element is an array of zeros and ones splitting the background and foreground.
        h: (integer) number of pixels on the vertical axis in each frame.
        w: (integer) number of pixels on the horizontal axis in each frame.
        l: (integer) number of frames per clip (SequenceExample).
    Return:
        couple of tensors containing the bounding boxes and the relative losses of the clip.
    """
    img_masks, loss_masks = [], []

    for i in range(l):
        # generate image mask:
        img_mask = img_mask_list[i]
        img_mask = tf.cast(tf.image.decode_png(img_mask), tf.float32)
        img_mask = tf.reshape(img_mask, (h, w))
        img_masks.append(img_mask)

        # generate loss mask:
        s_total = h * w
        s_mask = tf.reduce_sum(img_mask)

        def f1(): return img_mask * ((s_total - s_mask) / s_mask - 1) + 1

        def f2(): return tf.zeros_like(img_mask)

        def f3(): return tf.ones_like(img_mask)

        loss_mask = tf.case([(tf.equal(s_mask, 0), f2), (tf.less(s_mask, s_total / 2), f1)], default=f3)
        loss_masks.append(loss_mask)

    return tf.stack(img_masks), tf.stack(loss_masks)


def parse_fn(serialized):
    # Define a dict with the data-names and dtypes we expect to find in the TFRecords file.
    context_features = {'height': tf.FixedLenFeature([], dtype=tf.int64),
                        'width': tf.FixedLenFeature([], dtype=tf.int64),
                        'sequence_length': tf.FixedLenFeature([], dtype=tf.int64),
                        'filename': tf.FixedLenFeature([], dtype=tf.string),
                        'm_label': tf.FixedLenFeature([], dtype=tf.int64),
                        'fg_label': tf.FixedLenFeature([], dtype=tf.int64)}

    # Add background features in case dataset contains it.
    if FLG.with_bg and 'bg_label' not in context_features:
        bg_features = {'bg_label': tf.FixedLenFeature([], dtype=tf.int64)}
        context_features.update(bg_features)

    sequence_features = {'frames': tf.FixedLenSequenceFeature([], dtype=tf.string),
                         'masks': tf.FixedLenSequenceFeature([], dtype=tf.string)}

    # Parse the serialized data so we get a dict with our data.
    context_parsed, sequence_parsed = tf.parse_single_sequence_example(serialized=serialized,
                                                                       context_features=context_features,
                                                                       sequence_features=sequence_features)
    frames = decode_clip(sequence_parsed['frames'], FLG.height, FLG.width, FLG.example_length)
    m_label = context_parsed['m_label']
    fg_label = context_parsed['fg_label']
    return frames, m_label, fg_label


def get_input_fn(filenames, train):
    """ Define the appropriate function to generate the input pipeline according to the specified mode.
    Args:
        filenames: (string) directory path for the TFRecords files.
        train: (boolean) whether training (True) or evaluating (False).
    Return:
        call to input_fn with the specified arguments.
    """

    def input_fn():
        """ Perform a set of transformations on the TFRecords files to feed
            the model in each step in an efficient and adequate way.
        Return:
             x: (dictionary) containing a batch of clips.
             y: (list of integer) containing a batch of labels.
                Each integer identifies the ground truth label id of the clip.
        """
        # Create a list with all the TFRecord files from the filenames directory.
        Files = tf.data.Dataset.list_files(filenames)

        # To start an input pipeline, we must define a source. In this case
        # we create a TensorFlow Dataset object which has functionality
        # for reading data from TFRecords files.
        Dataset = tf.data.TFRecordDataset(filenames=Files)

        # -------- Perform some transformations to Dataset -------- #

        if train:
            # Read a buffer of the given size and randomly shuffle it.
            # Repeat the data according to the number of epochs.
            Dataset = Dataset.apply(tf.contrib.data.shuffle_and_repeat(FLG.train_length, FLG.checkpoint_epochs))
        else:
            # If evaluating then don't shuffle the data. Only go through the data once.
            Dataset = Dataset.repeat(1)

        Dataset = Dataset.apply(tf.contrib.data.map_and_batch(map_func=parse_fn,
                                                              num_parallel_batches=FLG.parallel_calls,
                                                              batch_size=FLG.batch_size))
        Dataset = Dataset.prefetch(1)

        # Create an Iterator for Dataset after the previous transformations.
        Iterator = Dataset.make_one_shot_iterator()

        # Get the next batch of clips, masks and labels.
        frames_batch, m_label_batch, fg_label_batch = Iterator.get_next()

        # The input_fn must return a dict wrapping the feature tensors and another warping the label
        # tensors because the Estimator API expects the input to the model_fn to be a dictionary.
        x = {'frames': frames_batch}
        y = {'m_label': m_label_batch, 'fg_label': fg_label_batch}
        return x, y
    return input_fn


# =============================== Feature columns =============================== #


def get_feature_columns():
    """ Map each 'feature' from the input_fn with a feature column to feed the model.
    Return:
        feature_columns: (list) each element of the list is a feature column.
    """
    feature_frames = tf.feature_column.numeric_column('frames', shape=[FLG.example_length,
                                                                       FLG.height,
                                                                       FLG.width,
                                                                       FLG.channel])
    feature_columns = [feature_frames]
    return feature_columns


# =============================== Custom Estimator ============================== #


def bn(x, training, activation=tf.nn.relu):
    """ Custom layer performing a batch normalization
    Args:
        x: (tensor) activation volume from the previous layer.
        training: (boolean) whether the network is in train (True) or validate (False) mode.
        activation: kind of activation function applied at the output of the layer.
    Return:
        tensor normalized in terms of batch.
    """
    return tf.contrib.layers.batch_norm(x, is_training=training, activation_fn=activation, **FLG.bn_params)


def conv_fn(x, name, params, is_training, first_conv=False, last_conv=False):
    """ Custom layer performing a 2D spatial convolution.
    Args:
        x: (tensor) activation volume from the previous layer.
        name: (string) name of the layer.
        params: (dictionary) all the parameters required to create a 2D convolutional layer.
        is_training: (boolean) whether compute the batch normalization (True) or not (False).
        first_conv: (boolean) if True, then the residual connections are not applied.
        last_conv: (boolean) if True, then the activation function is not computed.
    Return:
        conv: (tensor) resulting from the convolution of the input tensor with the filterbank of the layer.
    """
    if 'size' in params:
        del params['size']
    aux_conv = params.copy()
    aux_conv['strides'] = 1

    if last_conv:
        conv = bn(tf.layers.conv2d(x, name=name + '_0', **aux_conv), is_training, None)
    else:
        conv = bn(tf.layers.conv2d(x, name=name + '_0', **aux_conv), is_training)
        conv = bn(tf.layers.conv2d(conv, name=name + '_1', **params), is_training)

    if first_conv:
        return conv
    else:
        aux_res = params.copy()
        aux_res['kernel_size'] = 1
        res = tf.layers.conv2d(x, **aux_res)
        conv += res
        return conv


def deconv_fn(x, name, params, is_training, last_deconv=False):
    """ Custom layer performing a 2D spatial deconvolution based on NN-Resize + conv2d layers.
    Args:
        x: (tensor) activation volume from the previous layer.
        name: (string) name of the layer.
        params: (dictionary) all the parameters required to create a 2D convolutional layer.
        is_training: (boolean) whether compute the batch normalization (True) or not (False).
        last_deconv: (boolean) if True, then the activation function is not computed.
    Return:
        conv: (tensor) resulting from the resize of the input tensor followed by a 2d convolution.
    """
    activation = tf.nn.relu

    if last_deconv:
        activation = None
    resize = tf.image.resize_images(images=x, size=params['size'], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    conv = bn(tf.layers.conv2d(inputs=resize, name=name,
                               padding=params['padding'],
                               filters=params['filters'],
                               kernel_size=params['kernel_size']), is_training, activation)
    return conv


def combine_features(x, y):
    """ Function that combines a_feat and m_feat tensors by multiplying them.
    Args:
        x: (tensor) appearance features.
        y: (tensor) motion features.
    Return:
        out: (tensor) product of both kind of features.
    """
    out = x*y
    return out


def dense_fn(x, name, is_training):
    """ Custom layer combining fully connected + dropout layers
    Args:
        x: (tensor) activation volume from the previous layer.
        name: (string) name of the layer.
        is_training: (boolean) whether compute the dropout (True) or not (False).
    Return:
         dense: (tensor) latent representation wrt appearance features.
    """
    dense = bn(tf.layers.dense(inputs=x, units=FLG.units, name=name), is_training)
    if is_training:
        return tf.nn.dropout(dense, FLG.dropout_rate)
    else:
        return dense


def classifier_fn(x, name, is_training, last_dense=False, units=FLG.units_classifier):
    """ Custom layer combining fully connected + dropout layers
    Args:
        x: (tensor) activation volume from the previous layer.
        name: (string) name of the layer.
        is_training: (boolean) whether compute the dropout (True) or not (False).
        last_dense:  (boolean) if True, then the activation function is not computed.
        units: (integer) Number of neurons used
    Return:
         dense: (tensor) output from the fully connected layer
    """
    if last_dense:
        return bn(tf.layers.dense(inputs=x, units=units, name=name), is_training, None)
    else:
        dense = bn(tf.layers.dense(inputs=x, units=units, name=name), is_training)
        if is_training:
            return tf.nn.dropout(dense, FLG.dropout_rate)
        else:
            return dense


def l2_norm(x):
    l2_norm_tensor = tf.sqrt(tf.reduce_sum(tf.pow(x, 2)))
    return l2_norm_tensor


def rgb2grayscale(rgb_input):
    gray_frames = []
    for i in range(rgb_input.shape[0]):
        gray_image = tf.squeeze(tf.image.rgb_to_grayscale(rgb_input[i]))
        gray_image = tf.image.convert_image_dtype((gray_image + 1) / 2, tf.uint8)
        gray_frames.append(gray_image)
    return gray_frames


def rgb2binary(rgb_input):
    gray_frames = rgb2grayscale(rgb_input)
    binary_frames = []
    for i in range(rgb_input.shape[0]):
        binary_image = tf.where(gray_frames[i] > 128, 255*tf.ones_like(gray_frames[i]), tf.zeros_like(gray_frames[i]))
        binary_frames.append(binary_image)
    return binary_frames


def get_labels(x):
    reshaped_tensor = tf.reshape(x, [FLG.batch_size*(FLG.seq_length-1), FLG.height, FLG.width, FLG.channel])
    labels = []
    for i in range(reshaped_tensor.shape[0]):
        image_labels = tf.image.convert_image_dtype((reshaped_tensor[i] + 1) / 2, tf.uint8)
        labels.append(image_labels)

    reshaped_labels = tf.reshape(labels, [FLG.batch_size, FLG.seq_length-1, FLG.height, FLG.width, FLG.channel])
    final_labels = tf.cast(reshaped_labels, tf.int64)
    return final_labels


def get_gt(x):
    gt = []
    for i in range(x.shape[0]):
        image_gt = tf.squeeze(tf.image.convert_image_dtype((x[i] + 1) / 2, tf.uint8))
        gt.append(image_gt)
    return tf.cast(gt, tf.float32)


def encode_gif(im_seq, tag, fps):
    """
    Given a 4D numpy tensor of images, encodes as a gif.
    """
    with tempfile.NamedTemporaryFile() as f:
        fname = f.name + '.gif'
    clip = mpy.ImageSequenceClip(list(im_seq), fps=fps)
    clip.write_gif(fname, verbose=False, progress_bar=False)

    with open(fname, 'rb') as f:
        enc_gif = f.read()
    os.remove(fname)

    # create a tensorflow image summary protobuf:
    im_summ = tf.Summary.Image()
    im_summ.height = im_seq.shape[1]
    im_summ.width = im_seq.shape[2]
    im_summ.colorspace = 3
    im_summ.encoded_image_string = enc_gif

    # create a summary obj:
    summ = tf.Summary()
    summ.value.add(tag=tag, image=im_summ)
    summ_str = summ.SerializeToString()
    return summ_str


def gif_summary(name, im_seq, fps, collections=None, family=None):
    """
    im_seq: 4D tensor (TxHxWxC) for which GIF is to be generated.
    collections: collections to which the summary op is to be added.
    """
    if summary_op_util.skip_summary():
        return constant_op.constant('')

    with summary_op_util.summary_scope(name, family, values=[im_seq]) as (tag, scope):
        gif_summ = tf.py_func(encode_gif, [im_seq, tag, fps], tf.string, stateful=False)
        summary_op_util.collect(gif_summ, collections, [tf.GraphKeys.SUMMARIES])

    return gif_summ
