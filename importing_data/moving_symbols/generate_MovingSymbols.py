# --------------------------------------------------------------------------- #
# --------------- Generate customized Moving Symbols dataset ---------------- #
# --------------------------------------------------------------------------- #

# ============================ Importing modules ============================ #

import cv2
import os
import numpy as np
import multiprocessing
import sys

# Import project modules
from moving_symbols import MovingSymbolsEnvironment, Symbol

# =============================== Parameters =============================== #


class FLG:
    def __init__(self):
        pass

# ------------------ Directory parameters ------------------ #

FLG.dataset_path = '/netscratch/arenas/dataset/moving_symbols'  # Working directory.
FLG.dataset_output = 'output/MovingSymbols2_Seen'               # Directory in which the generated dataset is stored.

# ------------------- Dataset parameters ------------------- #

FLG.with_bg = False                         # It determines whether the dataset has background or not.
FLG.images = 'mnist'                        # Images used as moving objects.
FLG.movements = ['Vertical', 'Horizontal']  # Kind of movements that the objects perform.

# Train variables:
FLG.num_train_videos = 5000          # Number of training clips per each movement.
FLG.num_train_frames = 20            # Total number of frames per clip.
FLG.train_labels = [0, 3]            # List containing the object labels.
FLG.train_size = (1, 1)              # Value ranges of the scaling factor applied to the original object.
FLG.train_speed = (8, 8)             # Value ranges of the #pixels the object is displaced between frames.
FLG.train_movements = FLG.movements  # Kind of movements applied to the training set.

# Test variables:
FLG.num_test_videos = 500
FLG.num_test_frames = 20
FLG.test_labels = [0, 3]
FLG.test_size = (1, 1)
FLG.test_speed = (8, 8)
FLG.test_movements = FLG.movements


class MovingSymbolsClassTrajectoryTracker:
    """Object that gets the symbol classes and trajectories of the generated video"""
    def __init__(self):
        self.symbol_classes = {}
        self.images = {}
        self.scales = {}
        self.trajectories = {}
        self.background_classes = []

    def process_message(self, message):
        """Store the message."""
        meta = message['meta']
        if message['type'] == 'background':
            self.background_classes = meta['label']
        elif message['type'] == 'symbol_init':
            self.symbol_classes[meta['symbol_id']] = meta['label']
            self.images[meta['symbol_id']] = meta['image']
        elif message['type'] == 'symbol_state':
            if meta['symbol_id'] not in self.trajectories:
                self.scales[meta['symbol_id']] = []
                self.trajectories[meta['symbol_id']] = []
            self.scales[meta['symbol_id']].append(meta['scale'])
            self.trajectories[meta['symbol_id']].append(meta['position'])

    def get_info(self):
        """
        Return:
            background_classes_np: (array) background label of the clip.
            symbol_classes_np: (array) symbols labels within the clip with shape=(S).
            images_np: (array) the symbol's full image with shape=(S, Hs, Ws, C).
            scales_np: (array) the symbol's scale with shape=(S, T).
            trajectories_np: the symbol's coordinates with shape=(S, T, 2).
        """
        background_classes_np = np.array([self.background_classes])
        sorted_keys = sorted(self.symbol_classes.keys())
        symbol_classes_np = np.array([self.symbol_classes[k] for k in sorted_keys])
        images_np = np.array([self.images[k] for k in sorted_keys])
        for k in sorted_keys:
            self.scales[k] = np.stack(self.scales[k], axis=0)
            self.trajectories[k] = np.stack(self.trajectories[k], axis=0)
        scales_np = np.stack([self.scales[k] for k in sorted_keys], axis=0)
        trajectories_np = np.stack([self.trajectories[k] for k in sorted_keys], axis=0)

        return background_classes_np, symbol_classes_np, images_np, scales_np, trajectories_np


def im2vid(vid_path, im_seq, mask_seq=None):
    """ Convert an image sequence to video and write to vid_path.
        If mask is given, then it generates a mask laid over video.
    Args:
        vid_path: (string) directory in which the generated dataset is stored.
        im_seq:   (array) all the frames within a clip with the pixel values in range [0, 255]. shape=(T, Hv, Wv, C).
        mask_seq: (array) same size and shape as im_seq: {0,1}: 0=BG, 1=FG: uint8
    """
    writer = cv2.VideoWriter(vid_path, cv2.VideoWriter_fourcc(*'MJPG'), 10, (im_seq[0].shape[1], im_seq[0].shape[0]))
    if not writer.isOpened():
        print('Video could not be written. Some bug!')
        return None
    for i in range(im_seq.shape[0]):
        if mask_seq is not None:
            mask = mask_seq[i]
            grayscaleimage = cv2.cvtColor(im_seq[i, :, :, ::-1].copy(), cv2.COLOR_BGR2GRAY)
            maskedimage = np.zeros(im_seq[i].shape, dtype=np.uint8)
            for c in range(3):
                maskedimage[:, :, c] = grayscaleimage / 2 + 127
            maskedimage[mask.astype(np.bool), :2] = 0
        else:
            maskedimage = im_seq[i, :, :, ::-1]
        writer.write(maskedimage)
    writer.release()


def generate_video_mask(symbols, video, num_frames, scales, trajectories):
    """ Create the bounding box of the moving symbol per each frame within the clip.
    Args:
        symbols: (array) the symbol's full image with shape=(S, Hs, Ws, C).
        video:   (array) all the frames within a clip with the pixel values in range [0, 255]. shape=(T, Hv, Wv, C).
        num_frames: (integer) total number of frames per clip (T).
        scales: (array) the symbol's scale.
        trajectories: (array) the symbol's coordinates.
    Return:
         mask: (array) bounding box of the moving symbol per each frame within the clip.
    """
    num_symbols = symbols.shape[0]  # number of symbols (S) within the clip.
    video_height = video.shape[1]   # number of pixels on the vertical axis in each frame (Hv).
    video_width = video.shape[2]    # number of pixels on the horizontal axis in each frame (Wv).

    # Create an empty mask (all zeros), in which the bounding box is generated by going through
    # all the positions of all the frames according to the size and position of the symbols.
    mask = np.zeros((num_symbols, num_frames, video_height, video_width))
    for s in range(num_symbols):
        # Obtain the height and width of each symbol.
        symbol_height = symbols[s].shape[0]  # number of pixels on the vertical axis of the symbol s (Hs).
        symbol_width = symbols[s].shape[1]   # number of pixels on the horizontal axis of the symbol s (Ws).
        for t in range(num_frames):
            # Locate the coordinates of the top left vertex of each symbol within the frame t.
            y_vertice = int(-trajectories[s][t][1] - (symbol_height * scales[s][t]) / 2)
            x_vertice = int(trajectories[s][t][0] - (symbol_width * scales[s][t]) / 2)
            for h in range(int(symbol_height * scales[s][t])):
                for w in range(int(symbol_width * scales[s][t])):
                    # When the symbol touches one of the limits of
                    # the frame, the value of the mask is not modified.
                    if abs(y_vertice + h) >= video_height or abs(x_vertice + w + 1) >= video_width:
                        continue
                    # Otherwise, the value of the mask in that position is one.
                    else:
                        mask[s][t][y_vertice + h - 1][x_vertice + w + 1] = 1
    return mask


def generate_moving_symbols_video((seed, num_frames, params)):
    """ Create the NumPy array for one video and its mask.
    Return:
        video_tensor: (array) all the frames within a clip with the pixel values in range [0, 255].
        masks: (array) bounding box of the moving symbol per each frame within the clip.
        symbol_classes: (integer) class that each symbol in the clip belongs.
        background_classes: (integer) class of the background in case of use it.
    """
    sub = MovingSymbolsClassTrajectoryTracker()
    env = MovingSymbolsEnvironment(params, seed)
    env.add_subscriber(sub)

    all_frames = []
    for _ in xrange(num_frames):
        frame = env.next()
        all_frames.append(np.array(frame))
    video_tensor = np.array(all_frames, dtype=np.uint8)
    background_classes, symbol_classes, symbols, scales, trajectories = sub.get_info()
    masks = generate_video_mask(symbols, video_tensor, num_frames, scales, trajectories)

    return video_tensor, masks, symbol_classes, background_classes


def generate_all_moving_symbol_videos(pool, pool_seed, num_videos, num_frames, params, dataset_name, output_dir):
    """ Create the NumPy array for all the videos and it masks.
    Args:
        pool: (instance of multiprocessing.Pool)
        pool_seed:  (integer) define a specific seed to perform the pooling.
        num_videos: (integer) number of clips per each movement.
        num_frames: (integer) total number of frames per clip.
        params: (dictionary) information required to generate the clips within the dataset.
        dataset_name: (string) kind of movement.
        output_dir:   (string) directory in which the generated dataset is stored.
    """
    print('Working on {}...'.format(dataset_name))
    arg_tups = [(seed, num_frames, params) for seed in xrange(pool_seed, pool_seed+num_videos)]
    video_data = pool.map(generate_moving_symbols_video, arg_tups)
    videos, masks, symbol_classes, background_classes = zip(*video_data)

    videos = np.stack(videos, axis=0)   # V x T x H x W x C
    masks = np.stack(masks, axis=0)     # V x T x H x W
    symbol_classes = np.stack(symbol_classes, axis=0)
    background_classes = np.stack(background_classes, axis=0)

    np.save(os.path.join(output_dir, '{}_videos.npy'.format(dataset_name)), videos)
    np.save(os.path.join(output_dir, '{}_masks.npy'.format(dataset_name)), masks)
    np.save(os.path.join(output_dir, '{}_symbol_classes.npy'.format(dataset_name)), symbol_classes)
    np.save(os.path.join(output_dir, '{}_background_classes.npy'.format(dataset_name)), background_classes)

    # Generate videos.avi (optional)
    for vid_id in range(videos.shape[0]):
        video = videos[vid_id, :]
        # mask = np.squeeze(masks[vid_id, :], axis=0)
        im2vid(os.path.join(output_dir, '{}_video_{}.avi'.format(dataset_name, vid_id+1)), video, mask_seq=None)


def get_param_dicts(images, split, symbol_labels, size, speed, movements, num_frames):
    """ Create a dictionary that collects all the required parameters entered as arguments.
    Args:
        images: (string) images used as moving objects.
        split:  (string) whether the parameters correspond to the training or the testing set.
        symbol_labels: (list of integers) containing the object labels.
        size:  (list of integers) value ranges of the scaling factor applied to the original object.
        speed: (list of integers) value ranges of the #pixels the object is displaced between frames.
        movements:  (list of strings) kind of movements that the objects perform.
        num_frames: (integer) Total number of frames per clip.
    Return:
        param_dicts: (dictionary) each key is a kind of movement
        and its value is a dictionary containing the information
        required to generate the clips within the dataset.
    """
    param_dicts = {}

    for index in movements:
        param_dicts[index] = {'data_dir': os.path.join(FLG.dataset_path, 'data', images),
                              'split': split,
                              'symbol_labels': symbol_labels,
                              'scale_limits': size,
                              'position_speed_limits': speed,
                              'lateral_motion_at_start': True,
                              'movement': index,
                              'video_len': num_frames}

        # In case the movement is Forward or Backward it is necessary to include new parameters and modify others.
        if index == 'Forward' or index == 'Backward':
            param_dicts[index].update({'position_speed_limits': (0, 0),
                                       'lateral_motion_at_start': False,
                                       'scale_function_type': 'triangle',
                                       'scale_period_limits': (num_frames, num_frames)})

        # In case we want to use another background (default is black) we need to specify
        # the directories where the background images are stored and their labels.
        if FLG.with_bg:
            bg_dicts = {'background_data_dir': os.path.join(dataset_path, 'data', 'sun_bg'),
                        'background_labels': os.listdir(os.path.join(dataset_path, 'data', 'sun_bg', split))}
            param_dicts[index].update(bg_dicts)

    return param_dicts


def main():
    FLG.pool_seed = 123
    FLG.pool = multiprocessing.Pool()

    # Create a dictionary with the required training parameters
    # to generate all the clips within the training set.
    train_params = get_param_dicts(FLG.images, 'training', FLG.train_labels, FLG.train_size,
                                   FLG.train_speed, FLG.train_movements, FLG.num_train_frames)

    # Create a dictionary with the required testing parameters
    # to generate all the clips within the testing set.
    test_params = get_param_dicts(FLG.images, 'testing', FLG.test_labels, FLG.test_size,
                                  FLG.test_speed, FLG.test_movements, FLG.num_test_frames)

    for dataset_name, params in train_params.iteritems():

        # Make output directory in which the generated training set is stored (in case is not created).
        train_dir = os.path.join(FLG.dataset_path, FLG.dataset_output, 'train', dataset_name)
        if not os.path.isdir(train_dir):
            os.makedirs(train_dir)

        generate_all_moving_symbol_videos(FLG.pool, FLG.pool_seed, FLG.num_train_videos, FLG.num_train_frames,
                                          params, dataset_name, train_dir)

    for dataset_name, params in test_params.iteritems():

        # Make output directory in which the generated testing set is stored (in case is not created).
        test_dir = os.path.join(FLG.dataset_path, FLG.dataset_output, 'test', dataset_name)
        if not os.path.isdir(test_dir):
            os.makedirs(test_dir)

        generate_all_moving_symbol_videos(FLG.pool, FLG.pool_seed, FLG.num_test_videos, FLG.num_test_frames,
                                          params, dataset_name, test_dir)

if __name__ == '__main__':
    main()
