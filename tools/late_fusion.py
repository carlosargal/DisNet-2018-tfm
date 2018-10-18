import os
import json
import pickle
import tensorflow as tf
import tensorflow.contrib.eager as tfe
import numpy as np
tfe.enable_eager_execution()

dir_path = '/netscratch/arenas/model/prob'
spatial_file = 'spatial_val_prob.pickle'
temporal_file = 'temporal_val_prob.pickle'
motion_file = 'motion_val_prob.pickle'


def pickle_read(directory, file):
    prob_dict = {}
    with (open(os.path.join(directory, file), "rb")) as openfile:
        while True:
            try:
                prob_dict.update(pickle.load(openfile))
            except EOFError:
                break

    return prob_dict

spatial_dict = pickle_read(dir_path, spatial_file)
temporal_dict = pickle_read(dir_path, temporal_file)
motion_dict = pickle_read(dir_path, motion_file)

# Three-stream average fusion
prob_list = []
label_list = []
average_prob = []
count = 0
for key, values in motion_dict.items():
    if key in spatial_dict.keys() and key in temporal_dict.keys():
        # Average element-wise probabilities
        arr = np.array([spatial_dict[key], temporal_dict[key], values[0]])
        average_prob = np.divide(arr.sum(axis=0), 3.0)
        prob_list.append(average_prob.tolist())
        # prob_list.append(values[0])
        label_list.append(values[1])
    else:
        count += 1
        # prob_list.append(values[0])
        # label_list.append(values[1])

pred = tf.one_hot(tf.nn.top_k(prob_list).indices, 101)
pred = tf.squeeze(pred, axis=1)
pred = tf.cast(pred, tf.bool)
labels = tf.cast(label_list, tf.bool)
ac = tf.reduce_sum(tf.cast(tf.logical_and(labels, pred), tf.float32)) / len(prob_list)
accuracy = ac.numpy().tolist() * 100

print('Number of mismatched videos: %s' % count)
print('Accuracy: %s' % accuracy)

