import os
import json
import pickle
import tensorflow as tf
import tensorflow.contrib.eager as tfe
import numpy as np
tfe.enable_eager_execution()

file_path = '/netscratch/arenas/model/prob/spatial_video_preds.pickle'
prob_save_dir = '/netscratch/arenas/model/prob'
prefix = 'spatial_val_prob'

count = 0
log_dict = {}
with (open(file_path, "rb")) as openfile:
    while True:
        try:
            log_dict.update(pickle.load(openfile))
        except EOFError:
            break

# Create prob output folder
if not os.path.exists(prob_save_dir):
    os.makedirs(prob_save_dir)
prob_file_json = open(os.path.join(prob_save_dir, prefix + '.json'), 'w')
prob_file_pickle = open(os.path.join(prob_save_dir, prefix + '.pickle'), 'wb')

prob_dict = {}
for key, logits in log_dict.items():
    count += 1
    logits_list = logits.tolist()
    prob = tf.nn.softmax(logits_list)
    prob_list = prob.numpy().tolist()
    new_key = 'v_' + key + '.avi'
    prob_dict[new_key] = prob_list

prob_file_json.write(json.dumps(prob_dict))
pickle.dump(prob_dict, prob_file_pickle)
print('number of videos: %s' % count)
