# Note: check ranges. tf.decode_jpeg=[0,1], ffmpeg=[0,255] (JPEG encodes [0,255] uint8 images)

# --------------------------------------------------------------------------- #
# ------------- Convert MovingSymbols dataset to TFRecord files ------------- #
# --------------------------------------------------------------------------- #

# ============================ Importing modules ============================ #

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import re
import sys
import time
import ipdb
import glob
import random
import os.path
import threading

from datetime import datetime

import numpy as np
import scipy.misc as sm
import tensorflow as tf
import xml.etree.ElementTree as et

# Include project root directory
sys.path.append(os.path.abspath('..'))

# Import project modules
from tools.ffmpeg_reader import decode_video


# =============================== Parameters =============================== #


class FLG:
    def __init__(self):
        pass

# ------------------ Directory parameters ------------------ #

storage_path = '/netscratch/arenas/dataset/moving_symbols'

FLG.output_directory = '/netscratch/arenas/tfrecords/MovingSymbols2-Seen-tf-records/test'  # Output data directory.
FLG.videos_directory = os.path.join(storage_path, 'output/MovingSymbols2_Seen/test')       # Video dataset directory.
FLG.input_file = os.path.join(storage_path, 'MovingSymbols2_testlist.txt')         # .txt with video filenames.
FLG.m_class_list = os.path.join(storage_path, 'MovingSymbols2_m_classlist.txt')    # .txt with motion class names.
FLG.fg_class_list = os.path.join(storage_path, 'MovingSymbols2_fg_classlist.txt')  # .txt with foreground class names.
FLG.bg_class_list = os.path.join(storage_path, 'MovingSymbols_bg_classlist.txt')   # .txt with background class names.

# ------------------- Dataset parameters ------------------- #

FLG.name = 'Moving-Symbols-test'  # Name for the subset.
FLG.job_id = 0        # Job ID for the multi-job scenario. In range [0, num_jobs-1].
FLG.num_jobs = 1      # Number of jobs will process this dataset.
FLG.num_threads = 1   # Number of threads within each job to preprocess the videos.
FLG.num_shards = 18   # Number of shards. Each job will process num_shards/num_jobs shards.
FLG.label_offset = 1  # Offset for class IDs. Use 1 to avoid confusion with zero-padded elements
FLG.resize_h = 64     # Height after resize.
FLG.resize_w = 64     # Width after resize.
FLG.with_bg = False   # It determines whether the dataset has background or not.


def convert_to_sequential_example(filename, video_buffer, mask_buffer,
                                  m_label, fg_label, bg_label,
                                  height, width, sample_length):
    """Build a SequenceExample proto for an example.
    Args:
        filename: (string) path to a video file, e.g., '/path/to/example.avi'.
        video_buffer: (array) with the video frames, with dims [n_frames, height, width, n_channels].
        mask_buffer: (array) activity masks of video frames.
        m_label: (list of integer) each integer identifies the ground truth label id.
        fg_label: (list of integer) each integer identifies the ground truth label id.
        bg_label: (list of integer) each integer identifies the ground truth label id.
        height: (integer) image height in pixels.
        width: (integer) image width in pixels.
        sample_length: (integer) length of sampled clips from video, set to -1 if don't want sampling.
    Returns:
        SequentialExample proto.
    """
    # Get sequence length.
    full_length = len(video_buffer)
    assert len(video_buffer) == len(mask_buffer)

    example_list = []
    if sample_length == -1:
        sample_length = full_length
    num_clips = full_length // sample_length
    for i in range(num_clips):
        # Create SequenceExample instance
        example = tf.train.SequenceExample()

        # Context features (non-sequential features).
        example.context.feature['height'].int64_list.value.append(height)
        example.context.feature['width'].int64_list.value.append(width)
        example.context.feature['sequence_length'].int64_list.value.append(sample_length)
        example.context.feature['filename'].bytes_list.value.append(str.encode(filename))
        example.context.feature['m_label'].int64_list.value.append(m_label)
        example.context.feature['fg_label'].int64_list.value.append(fg_label)
        example.context.feature['bg_label'].int64_list.value.append(bg_label)

        # Sequential features.
        frames = example.feature_lists.feature_list['frames']
        masks = example.feature_lists.feature_list['masks']

        for j in range(sample_length):
            frames.feature.add().bytes_list.value.append(video_buffer[i*sample_length+j])  # .tostring())
            masks.feature.add().bytes_list.value.append(mask_buffer[i*sample_length+j])

        example_list.append(example)

    return example_list


class VideoCoder(object):
    """ Helper class providing TensorFlow image coding utilities """

    def __init__(self):
        # Create a single Session to run all image coding calls.
        # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.2)
        # self._sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        self._sess = tf.Session()

        # Initializes function that decodes video
        self._video_path = tf.placeholder(dtype=tf.string)
        self._decode_video = decode_video(self._video_path)

        # Initialize function to JPEG-encode a frame
        self._raw_frame = tf.placeholder(dtype=tf.uint8, shape=[None, None, 3])
        self._raw_mask = tf.placeholder(dtype=tf.uint8, shape=[None, None, 1])
        self._encode_frame = tf.image.encode_jpeg(self._raw_frame, quality=100)
        self._encode_mask = tf.image.encode_png(self._raw_mask)

    def decode_video(self, video_data):
        video, _, _, seq_length = self._sess.run(self._decode_video, feed_dict={self._video_path: video_data})
        raw_height, raw_width = video.shape[1], video.shape[2]
        assert len(video.shape) == 4
        assert video.shape[3] == 3
        return video, raw_height, raw_width, seq_length

    def encode_frame(self, raw_frame):
        return self._sess.run(self._encode_frame, feed_dict={self._raw_frame: raw_frame})

    def encode_mask(self, raw_mask):
        return self._sess.run(self._encode_mask, feed_dict={self._raw_mask: raw_mask})


def _resize_bbx(parsed_bbx, frame_h, frame_w):
    ratio_h = FLG.resize_h / frame_h
    ratio_w = FLG.resize_w / frame_w
    parsed_bbx[:, :, 0] = parsed_bbx[:, :, 0] * ratio_h
    parsed_bbx[:, :, 1] = parsed_bbx[:, :, 1] * ratio_w
    parsed_bbx[:, :, 2] = parsed_bbx[:, :, 2] * ratio_w
    parsed_bbx[:, :, 3] = parsed_bbx[:, :, 3] * ratio_h
    return parsed_bbx


def _bbx_to_mask(parsed_bbx, num_frames, frame_h, frame_w):

    if not num_frames == parsed_bbx.shape[0]:
        # align frames and bbx.
        if num_frames > parsed_bbx.shape[0]:
            padding = np.zeros([num_frames-parsed_bbx.shape[0], parsed_bbx.shape[1], parsed_bbx.shape[2]])
            parsed_bbx = np.concatenate([parsed_bbx, padding])
        else:
            parsed_bbx = parsed_bbx[:num_frames]

    masks = np.zeros([num_frames, frame_h, frame_w, 1])
    num_objs = parsed_bbx.shape[1]

    for i in range(num_frames):
        for j in range(num_objs):
            bbx = parsed_bbx[i, j]
            h, w, x, y = bbx[0], bbx[1], bbx[2], bbx[3]
            x_ = int(np.clip(x+w, 0, frame_w))
            y_ = int(np.clip(y+h, 0, frame_h))
            x = int(np.clip(x, 0, frame_w-1))
            y = int(np.clip(y, 0, frame_h-1))
            masks[i, y:y_, x:x_] = 1

    return masks


def process_video(filename, coder):
    """ Process a single video file using FFmpeg.
    Args:
        filename: (string) path to the video file.
        coder: (instance of VideoCoder) provide TensorFlow image coding utils.
    Returns:
        encoded_frames_seq: (array) video frames collection.
        encoded_mask_seq: (array) activity mask of the video frames.
        frame_h: (integer) height of the frames in pixels.
        frame_w: (integer) width of the frames in pixels.
        seq_length: (integer) sequence length (non-zero frames).
    """

    video, raw_h, raw_w, seq_length = coder.decode_video(filename)
    video = video.astype(np.uint8)
    assert len(video.shape) == 4
    assert video.shape[3] == 3
    frame_h, frame_w = video.shape[1], video.shape[2]

    # generate mask from annotations
    masks_npy = np.load(os.path.join(FLG.videos_directory,
                                     filename.split('/')[-2],
                                     '{}_masks.npy'.format(filename.split('/')[-2])))

    num_video = filename.split('_')[-1]
    num_video = num_video.split('.')[0]
    mask_npy = masks_npy[int(num_video) - 1]
    final_mask = np.squeeze(mask_npy, axis=0)
    final_mask = np.expand_dims(final_mask, axis=3)

    seq_length = np.asscalar(np.array(final_mask.shape[0]))

    encoded_frames_seq = []
    encoded_mask_seq = []
    for idx in range(final_mask.shape[0]):
        encoded_frames_seq.append(coder.encode_frame(video[idx, :, :, :]))
        encoded_mask_seq.append(coder.encode_mask(final_mask[idx, :, :, :]))

    return encoded_frames_seq, encoded_mask_seq, frame_h, frame_w, seq_length


def process_video_files_batch(coder, thread_index, ranges, name, job_index, num_jobs, num_shards,
                              filenames, m_labels, fg_labels, bg_labels):
    """ Process and save list of videos as TFRecord in 1 thread.
    Args:
        coder: instance of VideoCoder to provide TensorFlow video coding utils.
        thread_index: integer, unique batch to run index is within [0, len(ranges)).
        ranges: list of pairs of integers specifying ranges of each batch to
                analyze in parallel.
        name: (string) unique identifier specifying the tfrecord names.
        job_index: (integer) unique job index in range [0, num_jobs-1].
        num_jobs: (integer) how many different jobs will process this data set.
        num_shards: (integer) number of shards for this data set.
        filenames: (list of strings) each string is a path to a video file.
        m_labels: (list of integer) each integer identifies the ground truth label id
        fg_labels: (list of integer) each integer identifies the ground truth label id
        bg_labels: (list of integer) each integer identifies the ground truth label id
    """

    # Be sure we can split all the shards into each job.
    assert not num_shards % num_jobs
    num_shards_per_job = num_shards / num_jobs

    # Each thread produces N shards where N = int(num_shards_per_job / num_threads).
    num_threads = len(ranges)
    assert not num_shards_per_job % num_threads
    num_shards_per_batch = int(num_shards_per_job / num_threads)

    shard_ranges = np.linspace(ranges[thread_index][0], ranges[thread_index][1], num_shards_per_batch + 1).astype(int)
    num_files_in_thread = ranges[thread_index][1] - ranges[thread_index][0]

    counter = 0
    for s in range(num_shards_per_batch):
        # Generate a sharded version of the file name, e.g. 'train-00002-of-00010'
        shard = thread_index * num_shards_per_batch + job_index * num_shards_per_job + s
        output_filename = '%s-%.5d-of-%.5d.tfrecord' % (name, shard, num_shards)
        output_file = os.path.join(FLG.output_directory, output_filename)
        writer = tf.python_io.TFRecordWriter(output_file)

        shard_counter = 0
        files_in_shard = np.arange(shard_ranges[s], shard_ranges[s + 1], dtype=int)
        for i in files_in_shard:
            filename = filenames[i]
            m_label = m_labels[i]
            fg_label = fg_labels[i]
            bg_label = bg_labels[i]

            video_buffer, mask_buffer, height, width, seq_length = process_video(filename, coder)

            if seq_length == 0:
                print('Skipping video with null length')
                continue

            example_list = convert_to_sequential_example(filename, video_buffer, mask_buffer,
                                                         m_label, fg_label, bg_label,
                                                         height, width, sample_length=20)
            for example in example_list:
                writer.write(example.SerializeToString())
            shard_counter += 1
            counter += 1

            if not counter % 100:

                print('{} [thread {}]: Processed {} of {} videos in thread batch.'
                      .format(datetime.now(), thread_index, counter, num_files_in_thread))
                sys.stdout.flush()

        print('{} [thread {}]: Wrote {} video chunks to {}'
              .format(datetime.now(), thread_index, shard_counter, output_file))
        sys.stdout.flush()

    print('{} [thread {}]: Wrote {} video chunks to {} shards.'
          .format(datetime.now(), thread_index, counter, num_files_in_thread))
    sys.stdout.flush()


def process_video_files(name, job_index, num_jobs, num_shards, filenames, m_labels, fg_labels, bg_labels):
    """
    Process and save list of videos as TFRecord of Example protos.
    Args:
        name: (string) unique identifier specifying the tfrecord names.
        job_index: (integer) unique job index in range [0, num_jobs-1].
        num_jobs: (integer) how many different jobs will process this data set.
        num_shards: (integer) number of shards for this data set.
        filenames: (list of strings) each string is a path to a video file.
        m_labels: (list of integer) each integer identifies the ground truth label id
        fg_labels: (list of integer) each integer identifies the ground truth label id
        bg_labels: (list of integer) each integer identifies the ground truth label id
    """
    assert len(filenames) == len(m_labels)
    assert len(filenames) == len(fg_labels)
    assert len(filenames) == len(bg_labels)

    # Break all examples into batches in two levels: first for jobs, then for threads within each job
    num_files = len(filenames)
    num_files_per_job = int(num_files / num_jobs)
    first_file = job_index * num_files_per_job
    last_file = min(num_files, (job_index + 1) * num_files_per_job)

    print('Job #{} will process files in range [{},{}]'.format(job_index, first_file, last_file - 1))

    local_filenames = filenames[first_file:last_file]
    local_m_labels = m_labels[first_file:last_file]
    local_fg_labels = fg_labels[first_file:last_file]
    local_bg_labels = bg_labels[first_file:last_file]

    spacing = np.linspace(0, len(local_filenames), FLG.num_threads + 1).astype(np.int)
    ranges = []

    for i in range(len(spacing) - 1):
        ranges.append([spacing[i], spacing[i+1]])

    # Launch a thread for each batch.
    print('Launching %d threads for spacings: %s' % (FLG.num_threads, ranges))
    sys.stdout.flush()

    # Create a mechanism for monitoring when all threads are finished.
    coord = tf.train.Coordinator()

    # Create a generic TensorFlow-based utility for converting all image codings.
    coder = VideoCoder()

    threads = list()
    for thread_index in range(len(ranges)):
        args = (coder, thread_index, ranges, name, job_index, num_jobs, num_shards,
                local_filenames, local_m_labels, local_fg_labels, local_bg_labels)

        t = threading.Thread(target=process_video_files_batch, args=args)
        t.start()
        threads.append(t)

    # Wait for all the threads to terminate.
    coord.join(threads)
    print('{}: Finished writing all {} videos in data set.'.format(datetime.now(), len(local_filenames)))
    sys.stdout.flush()


def find_video_files(dataset_dir, input_file, m_class_list_path, fg_class_list_path, bg_class_list_path):
    """Build a list of all videos files and labels in the data set.
    Args:
        dataset_dir: (string) path to the dataset directory.
        input_file: (string) path to the text file listing.
        m_class_list_path: (string) path to the m_labels text file.
        fg_class_list_path: (string) path to the fg_labels text file.
        bg_class_list_path: (string) path to the bg_labels text file.
    Returns:
        filenames: (list of strings) each string is a path to a video file.
        m_labels: (list of integer) each integer identifies the ground truth label id
        fg_labels: (list of integer) each integer identifies the ground truth label id
        bg_labels: (list of integer) each integer identifies the ground truth label id
    """

    # Create a list with all the lines inside 'input_file'
    lines = [line.strip().split()[0] for line in open(input_file, 'r')]

    # Create a list with the motion classes inside 'm_class_list_path'
    m_list = [line.strip().split()[0] for line in open(m_class_list_path, 'r')]

    # Create a list with the foreground classes inside 'fg_class_list_path'
    fg_list = [line.strip().split()[0] for line in open(fg_class_list_path, 'r')]

    # Create a list with the background classes inside 'bg_class_list_path'
    if FLG.with_bg:
        bg_list = [line.strip().split()[0] for line in open(bg_class_list_path, 'r')]
    else:
        bg_list = []

    filenames = list()
    m_labels = list()
    fg_labels = list()
    bg_labels = list()

    for i, line in enumerate(lines):
        video = line
        num_video = line.split('_')[-1]
        num_video = num_video.split('.')[0]

        m_label_name = line.split("/")[0]
        m_label_id = m_list.index(m_label_name)

        fg_npy = np.load(os.path.join(dataset_dir, m_label_name, '{}_symbol_classes.npy'.format(m_label_name)))
        fg_label_name = fg_npy[int(num_video)-1][0]
        fg_label_id = fg_list.index(str(fg_label_name))

        bg_npy = np.load(os.path.join(dataset_dir, m_label_name, '{}_background_classes.npy'.format(m_label_name)))
        bg_label_name = bg_npy[int(num_video)-1]
        if FLG.with_bg:
            bg_label_id = bg_list.index(bg_label_name)
        else:
            bg_label_id = 0

        filenames.append(os.path.join(dataset_dir, video))
        m_labels.append(m_label_id)
        fg_labels.append(fg_label_id)
        bg_labels.append(bg_label_id)

    # Shuffle the ordering of all video files in order to guarantee random ordering of the images with respect to
    # label in the saved TFRecord files. Make the randomization repeatable.
    shuffled_index = list(range(len(filenames)))
    random.seed(12345)
    random.shuffle(shuffled_index)

    filenames = [filenames[i] for i in shuffled_index]
    m_labels = [m_labels[i] for i in shuffled_index]
    fg_labels = [fg_labels[i] for i in shuffled_index]
    bg_labels = [bg_labels[i] for i in shuffled_index]

    print('Found %d video files.' % len(filenames))

    return filenames, m_labels, fg_labels, bg_labels


def process_dataset(dataset_dir, input_file, m_class_list_path, fg_class_list_path, bg_class_list_path,
                    name, job_index, num_jobs, num_shards):
    """
    Process a complete data set and save it as TFRecord files.
    Args:
        dataset_dir: (string) path to the dataset directory.
        input_file: (string) path to the text file listing.
        m_class_list_path: (string) path to the m_labels text file.
        fg_class_list_path: (string) path to the fg_labels text file.
        bg_class_list_path: (string) path to the bg_labels text file.
        name: (string) unique identifier specifying the tfrecord names.
        job_index: (integer) unique job index in range [0, num_jobs-1].
        num_jobs: (integer) how many different jobs will process this data set.
        num_shards: (integer) number of shards for this data set.
    """
    filenames, m_labels, fg_labels, bg_labels = find_video_files(dataset_dir,
                                                                 input_file,
                                                                 m_class_list_path,
                                                                 fg_class_list_path,
                                                                 bg_class_list_path)

    process_video_files(name, job_index, num_jobs, num_shards, filenames, m_labels, fg_labels, bg_labels)


def main(_):
    # Be sure the number of jobs, threads and shards specified are multiples of each other.
    assert not int(FLG.num_shards / FLG.num_jobs) % FLG.num_threads, (
        'Please make the FLG.num_threads commensurate with FLG.num_shards and FLG.num_jobs')

    # Create a new output directory in case it does not exist.
    if not os.path.exists(FLG.output_directory):
        os.makedirs(FLG.output_directory)
    print('Saving results to {}'.format(FLG.output_directory))

    # Run it!
    process_dataset(FLG.videos_directory, FLG.input_file, FLG.m_class_list, FLG.fg_class_list, FLG.bg_class_list,
                    FLG.name, FLG.job_id, FLG.num_jobs, FLG.num_shards)


if __name__ == '__main__':
    tf.app.run(main)
