# --------------------------------------------------------------------------- #
# -------------------- Baseline: Vanilla frame prediction ------------------- #
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

# ------------------ Directory parameters ------------------ #

path_tfrecords_train = '/netscratch/arenas/tfrecords/MovingSymbols2-Seen-tf-records/train/*.tfrecord'
path_tfrecords_test = '/netscratch/arenas/tfrecords/MovingSymbols2-Seen-tf-records/test/*.tfrecord'
model_dir = '/netscratch/arenas/model/MovingSymbols2_Seen/MovingSymbols2_Seen_FramePredictor_bce'


# =============================== Custom Estimator ============================== #


def get_estimator(hparams, run_config):
    """ Define the appropriate function to generate the neural network according to the specified mode.
    Args:
        hparams: set of parameters used inside the model.
        run_config: set of parameters related with the creation of the model, such as directores, summaries and ckpts.
    Return:
        instance of an Estimator class to train, evaluate or predict on the model.
    """

    def model_fn(features, labels, mode, params):
        """ Create the computational graph of the model.
        Args:
            features: (dictionary or tensor) this is the first item returned from the input_fn.
            labels: (dictionary or tensor) this is the second item returned from the input_fn.
            mode: either TRAIN, EVAL, or PREDICT.
            params: (dictionary) user-defined hyper-parameters. Will receive what is passed to Estimator in params.
        Return:
            EstimatorSpec determining the behavior of the network based on the mode.
        """

        # Variable that controls when bn and dropout should be done.
        is_training = mode is tf.estimator.ModeKeys.TRAIN

        # ------------------------- Input ------------------------- #

        with tf.variable_scope('input'):
            # Get next batch of clips where values are in range [0, 1].
            input_layer = tf.feature_column.input_layer(features, params.feature_columns)
            input_layer = tf.reshape(input_layer, [FLG.batch_size,
                                                   FLG.example_length,
                                                   FLG.height,
                                                   FLG.width,
                                                   FLG.channel])

            # Binarize input_layer to take values 0 or 1.
            bin_input_layer = tf.round(input_layer)

            # Standardize input values to take values -1 or 1.
            std_input_layer = bin_input_layer * 2 - 1

        # ------------------------ Encoder ------------------------ #

        with tf.variable_scope('encoder'):
            # Spatial 2D conv whose weights are shared across all the frames:
            input_encoder = tf.reshape(std_input_layer[:, :-2], [-1, FLG.height, FLG.width, FLG.channel])

            # Input tensor: (batch_size x (example_length-2))x64x64x(#channels)
            conv = mf.conv_fn(input_encoder, 'conv1', FLG.conv1_params, is_training, first_conv=True)

            # Input tensor: (batch_size x (example_length-2))x32x32x(#filt1_encoder)
            conv = mf.conv_fn(conv, 'conv2', FLG.conv2_params, is_training)

            # Input tensor: (batch_size x (example_length-2))x16x16x(#filt2_encoder)
            conv = mf.conv_fn(conv, 'conv3', FLG.conv3_params, is_training)

            # Input tensor: (batch_size x (example_length-2))x8x8x(#filt3_encoder)
            conv = mf.conv_fn(conv, 'conv4', FLG.conv4_params, is_training)

            # Input tensor: (batch_size x (example_length-2))x4x4x(#filt4_encoder)
            conv = mf.conv_fn(conv, 'conv5', FLG.conv5_params, is_training)

            # Input tensor: (batch_size x (example_length-2))x2x2x(#filt5_encoder)
            conv = mf.conv_fn(conv, 'last_conv', FLG.last_conv_params, is_training, last_conv=True)

            # Input tensor: (batch_size x (example_length-2))x1x1x(#last_filt_encoder)
            dense = tf.reshape(conv, [-1, FLG.example_length-2, FLG.last_conv_params['filters']], name='dense')

            # Temporal 1D convolution:
            with tf.variable_scope('temporal_cnn'):
                # Input tensor: (batch_size)x(example_length-2)x(#last_filt_encoder)
                TCN = tcn.CausalConv1D(name='tcn', **FLG.temp_params)
                latent_space = mf.bn(TCN(dense), is_training)
                latent_space = tf.reshape(latent_space[:, FLG.seq_length-1:],
                                          [-1, FLG.last_conv_params['filters']],
                                          name='latent_space')

        # ------------------------ Decoder ------------------------ #

        with tf.variable_scope('decoder'):
            # Input tensor: (batch_size x (sequence_length-1))x(#last_filt_encoder)
            input_decoder = tf.reshape(latent_space,
                                       [FLG.batch_size*(FLG.seq_length-1), 1, 1, FLG.last_conv_params['filters']],
                                       name='input_decoder')

            # Input tensor: (batch_size x (sequence_length-1))x1x1x(#units a_feat)
            deconv = mf.deconv_fn(input_decoder, 'deconv1', FLG.deconv1_params, is_training)

            # Input tensor: (batch_size x (sequence_length-1))x2x2x(#filt1_decoder)
            deconv = mf.deconv_fn(deconv, 'deconv2', FLG.deconv2_params, is_training)

            # Input tensor: (batch_size x (sequence_length-1))x4x4x(#filt2_decoder)
            deconv = mf.deconv_fn(deconv, 'deconv3', FLG.deconv3_params, is_training)

            # Input tensor: (batch_size x (sequence_length-1))x8x8x(#filt3_decoder)
            deconv = mf.deconv_fn(deconv, 'deconv4', FLG.deconv4_params, is_training)

            # Input tensor: (batch_size x (sequence_length-1))x16x16x(#filt4_decoder)
            deconv = mf.deconv_fn(deconv, 'deconv5', FLG.deconv5_params, is_training)

            # Input tensor: (batch_size x (sequence_length-1))x32x32x(#filt5_decoder)
            deconv = mf.deconv_fn(deconv, 'last_deconv', FLG.last_deconv_params, is_training, last_deconv=True)

            # Input tensor: (batch_size x (sequence_length-1))x64x64x2
            output_decoder = tf.reshape(deconv,
                                        [FLG.batch_size,
                                         FLG.seq_length-1,
                                         FLG.height,
                                         FLG.width,
                                         FLG.last_deconv_params['filters']],
                                        name='output_decoder')

        # ---- Implement training, evaluation, and prediction ---- #

        # Provide an EstimatorSpec for ModeKeys.PREDICT mode:
        if mode == tf.estimator.ModeKeys.PREDICT:
            # Convert predicted_indices back into strings.
            predictions = {'encoding': latent_space, 'reconstruction': output_decoder}
            export_outputs = {'predict': tf.estimator.export.PredictOutput(predictions)}
            return tf.estimator.EstimatorSpec(mode, predictions=predictions, export_outputs=export_outputs)

        # Learning rate scheduler using exponential decay.
        global_step = tf.train.get_global_step()
        learning_rate = tf.train.exponential_decay(FLG.learning_rate,
                                                   global_step,
                                                   FLG.decay_steps,
                                                   FLG.decay_rate,
                                                   staircase=True)

        # Loss function based on reconstruction and regularization:
        with tf.variable_scope('loss'):
            binary_labels = tf.cast(bin_input_layer[:, FLG.seq_length:-1], tf.int64)
            loss = tf.losses.sparse_softmax_cross_entropy(labels=binary_labels, logits=output_decoder)
            loss += tf.losses.get_regularization_loss()

        # Optimization method:
        optimizer = tf.train.AdamOptimizer(learning_rate)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

        with tf.control_dependencies(update_ops):
            gradients = optimizer.compute_gradients(loss=loss)
            for gradient, variable in gradients:
                tf.summary.histogram('gradients/' + variable.name, mf.l2_norm(gradient))
                tf.summary.histogram('variables/' + variable.name, mf.l2_norm(variable))
            train_op = optimizer.apply_gradients(gradients, global_step=global_step)

        # ----------------------- Summaries ----------------------- #

            # Binary clip summaries:
            # Predicted frames [t+1, t+N-1] with binary pixel values (0 or 255).
            predicted_frames_bin = tf.squeeze(tf.cast(tf.argmax(output_decoder[0], axis=3)*255, tf.float32))

            # Ground truth frames [t+1, t+N-1] with binary pixel values (0 or 255).
            original_frames_bin = tf.squeeze(bin_input_layer[0, FLG.seq_length:-1]*255)

            # Ground truth frames [t, t+N-2] with binary pixel values (0 or 255).
            current_frames_bin = tf.squeeze(bin_input_layer[0, FLG.seq_length-1:-2]*255)

            comparative_frames_bin = tf.stack([predicted_frames_bin, original_frames_bin, original_frames_bin], 3)
            consecutive_frames_bin = tf.stack([current_frames_bin, original_frames_bin, original_frames_bin], 3)

            tf.summary.image('original_vs_predicted_bin', comparative_frames_bin, max_outputs=FLG.num_img_outputs)
            tf.summary.image('consecutive_frames_bin', consecutive_frames_bin, max_outputs=FLG.num_img_outputs)

            mf.gif_summary('original_vs_predicted_bin_gif', comparative_frames_bin, fps=FLG.fps)
            mf.gif_summary('consecutive_frames_bin_gif', consecutive_frames_bin, fps=FLG.fps)

        # Merge both rmse and original_vs_predicted summaries to be shown in EVAL mode.
        eval_summary_hook = tf.train.SummarySaverHook(save_steps=FLG.checkpoint_steps,
                                                      output_dir=model_dir + '/eval',
                                                      summary_op=tf.summary.merge_all())
        tf.summary.scalar('learning_rate', learning_rate)

        # Provide an estimator spec for ModeKeys.EVAL and ModeKeys.TRAIN modes:
        EstimatorSpec = tf.estimator.EstimatorSpec(mode=mode,
                                                   loss=loss,
                                                   train_op=train_op,
                                                   evaluation_hooks=[eval_summary_hook])
        return EstimatorSpec

    return tf.estimator.Estimator(model_fn=model_fn, model_dir=model_dir, params=hparams, config=run_config)


# ========================== Instance of the Estimator ========================== #


def main(_):
    if not os.path.exists(model_dir):
        print('Saving model to %s' % model_dir)
        os.makedirs(model_dir)

    # Create a HParams object specifying names and values of the model.
    HParams = tf.contrib.training.HParams(feature_columns=mf.get_feature_columns())

    # This class specifies the configurations for an Estimator run.
    RunConfig = tf.estimator.RunConfig(model_dir=model_dir,
                                       save_summary_steps=FLG.summary_steps,
                                       save_checkpoints_steps=FLG.checkpoint_steps,
                                       keep_checkpoint_max=FLG.num_checkpoints)

    # Instance of an Estimator class to train and evaluate TensorFlow models.
    Estimator = get_estimator(HParams, RunConfig)

    if not FLG.resume_training:
        print('Removing previous files from model_dir...')
        shutil.rmtree(model_dir)

    count = 1
    max_steps = 0
    while max_steps < FLG.train_steps:
        # Start training:
        print('Training...')
        max_steps = FLG.checkpoint_steps * count
        Estimator.train(max_steps=max_steps, input_fn=mf.get_input_fn(filenames=path_tfrecords_train, train=True))

        # Start evaluation:
        print('Evaluating...')
        Estimator.evaluate(steps=None, input_fn=mf.get_input_fn(filenames=path_tfrecords_test, train=False))

        count += 1

    print('Process completed successfully')


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main)
