# --------------------------------------------------------------------------- #
# -------------------------- Copy frame predictor --------------------------- #
# --------------------------------------------------------------------------- #

# ============================ Importing modules ============================ #

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import shutil
import tensorflow as tf

from tools.model_parameters import FLG   # Class containing all the required parameters to run the models.
from tools import model_functions as mf  # Collection of functions used by the models.
from tools import tcn                    # Temporal CausalConvnet1D.

# =============================== Parameters =============================== #


path_tfrecords = '/netscratch/arenas/tfrecords/MovingSymbols2-Seen-tf-records/train/*.tfrecord'


# ============================= Input pipeline ============================= #


def parse_fn(serialized):
    """ Define a dict with the data-names and dtypes we expect to find in the TFRecords file.
    Return:
         frames: (tensor) decoded clip within the value range [-1, 1].
    """
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
    frames = mf.decode_clip(sequence_parsed['frames'], FLG.height, FLG.width, FLG.example_length)
    return frames


def input_fn(filenames):
    """ Perform a set of transformations on the TFRecords files to feed
        the model in each step in an efficient and adequate way.
    Return:
         frames_batch: (dictionary) containing a batch of clips.
    """
    # Create a list with all the TFRecords files from the filenames directory.
    Files = tf.data.Dataset.list_files(filenames)

    # To start an input pipeline, we must define a source. In this case
    # we create a TensorFlow Dataset object which has functionality
    # for reading data from TFRecords files.
    Dataset = tf.data.TFRecordDataset(filenames=Files)

    # -------- Perform some transformations to Dataset: -------- #

    # Only go through the data once.
    Dataset = Dataset.repeat(1)

    Dataset = Dataset.apply(tf.contrib.data.map_and_batch(map_func=parse_fn,
                                                          num_parallel_batches=FLG.parallel_calls,
                                                          batch_size=FLG.batch_size))
    Dataset = Dataset.prefetch(1)

    # Create an Iterator for Dataset after the previous transformations.
    Iterator = Dataset.make_one_shot_iterator()

    # Get the next batch of clips.
    frames_batch = Iterator.get_next()
    return frames_batch


# =============================== Computational Graph ============================== #


def main(_):
    with tf.Session() as sess:
        # Call input_fn to get batches of frames in each step.
        frames_batch = input_fn(filenames=path_tfrecords)

        # Ensure all batches has the correct shape before computing the BCE.
        input_batch = tf.reshape(frames_batch, [-1, FLG.example_length, FLG.height, FLG.width, FLG.channel])

        # Binarize input_layer to take values 0 or 1.
        input_batch_bin = tf.round(input_batch)

        # Standardize input values to take values in range [-1, 1].
        std_input_batch = input_batch_bin * 2 - 1

        # Compute the BCE between each pair of consecutive frames
        # from 10th to 20th frame for all the training set.
        x = std_input_batch[:, FLG.seq_length-1:-2]  # Current frame
        y = input_batch_bin[:, FLG.seq_length:-1]  # Next frame
        q = tf.nn.sigmoid(x)
        loss_op = tf.reduce_mean(y * -tf.log(q) + (1 - y) * -tf.log(1 - q))

        init = tf.global_variables_initializer()
        sess.run(init)

        # Iterate over all the training set within the for loop,
        # computing the BCE per batch and storing it in loss_list.
        loss_list = []
        for i in range(int(FLG.steps_per_epoch)):
            loss_list.append(sess.run(loss_op))

        # Compute the final_loss (aka reference error) averaging
        # all the losses and print the result.
        final_loss = tf.reduce_mean(loss_list)
        final_loss = tf.Print(final_loss, [final_loss], 'The reference error in training set is: ')
        sess.run(final_loss)

        print('Process completed successfully')


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main)
