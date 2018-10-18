# --------------------------------------------------------------------------- #
# ------------------------- Disentangled Classifier ------------------------- #
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
AE_dir = '/netscratch/arenas/model/MovingSymbols2_Seen/MovingSymbols2_Seen_DisentangledFP_bce'
model_dir = '/netscratch/arenas/model/MovingSymbols2_Seen/MovingSymbols2_Seen_ClassifierDisFP_bce'


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

            # Binarize input_layer to take values 0 or 1. Uncomment only when is pre-trained with BCE model.
            input_layer = tf.round(input_layer)

            # Standardize input values to take values in range [-1, 1].
            std_input_layer = input_layer * 2 - 1

        # ------------------------ Encoder ------------------------ #

        with tf.variable_scope('encoder'):
            # Spatial 2D conv whose weights are shared across all the frames in range(example_length-2):
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

            # FC layer:
            with tf.variable_scope('fc'):
                # Input tensor: (batch_size)x(example_length-2)x(#last_filt_encoder)
                input_dense = tf.reshape(dense[:, FLG.seq_length - 1:],
                                         [-1, FLG.last_conv_params['filters']],
                                         name='input_dense')

                # Input tensor: (batch_size x (sequence_length-1))x(#last_filt_encoder)
                dense1 = mf.dense_fn(input_dense, 'dense1', is_training)

                # Input tensor: (batch_size x (sequence_length-1))x(#units a_feat)
                a_feat = mf.dense_fn(dense1, 'dense2', is_training)

            # Temporal 1D convolution:
            with tf.variable_scope('temporal_cnn'):
                # Input tensor: (batch_size)x(example_length-2)x(#last_filt_encoder)
                TCN = tcn.CausalConv1D(name='tcn', **FLG.temp_params)
                m_feat = mf.bn(TCN(dense), is_training)
                m_feat = tf.reshape(m_feat[:, FLG.seq_length-1:], [-1, FLG.last_conv_params['filters']], name='m_feat')

        # ----------------------- Classifier ---------------------- #

        with tf.variable_scope('classifier', reuse=tf.AUTO_REUSE):
            feat_dict = {'fg_label': a_feat, 'm_label': m_feat}
            dense_dict = {}
            for key in feat_dict.keys():
                # Input tensor: (batch_size x (sequence_length-1))x(#last_filt_encoder)
                dense4 = mf.classifier_fn(feat_dict[key], 'dense4', is_training)
                dense_dict.update({key: dense4})

        # ------------- Logits, labels and predictions ------------ #

        logits_dict = {}
        labels_dict = {}
        predictions_dict = {}

        for key, value in labels.items():
            with tf.variable_scope('logits_{}'.format(key), values=(dense_dict[key],)) as scope:
                # Input tensor: (batch_size x sequence_length)x(#units)
                # This is the last layer so it does not use an activation function.
                logits = mf.classifier_fn(dense_dict[key], scope, is_training, last_dense=True, units=FLG.num_class[key])

            # Convert the labels to one hot vector format.
            label = tf.one_hot(tf.cast(value, tf.int32), FLG.num_class[key])
            # Replicate the labels of each clip for the seq_length-1 frames predicted.
            label = tf.reshape(tf.tile(tf.expand_dims(label, 1), [1, FLG.seq_length-1, 1]),
                               [-1, FLG.num_class[key]])
            # Restore the labels to index format again with this new shape.
            label = tf.argmax(label, axis=1)

            # Predictions used to compute the PREDICT mode and the accuracy.
            predictions = {'classes': tf.argmax(logits, axis=1),
                           'probabilities': tf.nn.softmax(logits, name='softmax_tensor')}

            # Store the logits, labels and predictions for each task in their respective dictionaries.
            logits_dict.update({key: logits})
            labels_dict.update({key: label})
            predictions_dict.update({key: predictions})

        # -------- Implement TRAIN, EVAL and PREDICT modes -------- #

        # Provide an EstimatorSpec for ModeKeys.PREDICT mode:
        if mode == tf.estimator.ModeKeys.PREDICT:
            predict = {'class_ids': predictions_dict[:, 'classes'],
                       'probabilities': predictions_dict[:, 'probabilities'],
                       'logits': tf.stack(logits_dict)}
            return tf.estimator.EstimatorSpec(mode, predictions=predict)

        # Learning rate scheduler using exponential decay.
        global_step = tf.train.get_global_step()
        learning_rate = tf.train.exponential_decay(FLG.learning_rate,
                                                   global_step,
                                                   FLG.decay_steps,
                                                   FLG.decay_rate,
                                                   staircase=True)

        # Loss function based on cross-entropy between the output of
        # the neural network and the true labels for the input data:
        with tf.variable_scope('loss'):
            loss = 0.

            for key in labels_dict.keys():
                losses = tf.losses.sparse_softmax_cross_entropy(labels=labels_dict[key], logits=logits_dict[key])
                loss += losses

        # Load the variables from the last Autoencoder ckpt
        # that match with this graph (pretrain the network)
        tf.train.init_from_checkpoint(AE_dir, {'input/': 'input/', 'encoder/': 'encoder/'})

        # Optimization method:
        optimizer = tf.train.AdamOptimizer(learning_rate)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            classifier_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='classifier/')
            fg_logits_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='logits_fg_label/')
            m_logits_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='logits_m_label/')

            vars_list = [classifier_vars, fg_logits_vars, m_logits_vars]
            gradients = optimizer.compute_gradients(loss=loss, var_list=vars_list)
            for gradient, variable in gradients:
                tf.summary.histogram('gradients/' + variable.name, mf.l2_norm(gradient))
                tf.summary.histogram('variables/' + variable.name, mf.l2_norm(variable))
            train_op = optimizer.apply_gradients(gradients, global_step=global_step)

        # ----------------------- Summaries ----------------------- #

        # The accuracy is computed automatically and is updated during validation set.
        eval_metric_ops = {'val_motion_accuracy': tf.metrics.accuracy(labels=labels_dict['m_label'],
                                                                      predictions=predictions_dict['m_label']['classes']),
                           'val_appearance_accuracy': tf.metrics.accuracy(labels=labels_dict['fg_label'],
                                                                          predictions=predictions_dict['fg_label']['classes'])}

        # Compute again the accuracy to double check the previous one and compare with the train accuracy.
        m_accuracy = tf.reduce_mean(tf.cast(tf.equal(labels_dict['m_label'], predictions_dict['m_label']['classes']), tf.float32))
        fg_accuracy = tf.reduce_mean(tf.cast(tf.equal(labels_dict['fg_label'], predictions_dict['fg_label']['classes']), tf.float32))
        tf.summary.scalar('motion_accuracy', m_accuracy)
        tf.summary.scalar('appearance_accuracy', fg_accuracy)

        eval_summary_hook = tf.train.SummarySaverHook(save_steps=FLG.checkpoint_steps,
                                                      output_dir=model_dir + '/eval',
                                                      summary_op=tf.summary.merge_all())

        tf.summary.scalar('learning_rate', learning_rate)

        # Provide an estimator spec for ModeKeys.EVAL and ModeKeys.TRAIN modes:
        EstimatorSpec = tf.estimator.EstimatorSpec(mode=mode,
                                                   loss=loss,
                                                   train_op=train_op,
                                                   evaluation_hooks=[eval_summary_hook],
                                                   eval_metric_ops=eval_metric_ops)
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
