import glob
import os
import hashlib
import time
import argparse
from mkdir_p import mkdir_p

from PIL import Image

import numpy as np
from keras.layers.merge import concatenate
from keras.models import Model
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Merge, Input
from keras.layers import Conv2D
from keras.layers.normalization import BatchNormalization
from keras import optimizers
from keras import backend as K
from keras.callbacks import ModelCheckpoint, EarlyStopping

import matplotlib.pyplot as plt

TRACK_CODES = set(map(lambda s: s.lower(),
    ["ALL", "MR","CM","BC","BB","YV","FS","KTB","RRy","LR","MMF","TT","KD","SL","RRd","WS",
     "BF","SS","DD","DK","BD","TC"]))

def is_valid_track_code(value):
    value = value.lower()
    if value not in TRACK_CODES:
        raise argparse.ArgumentTypeError("%s is an invalid track code" % value)
    return value

OUT_SHAPE = 1

INPUT_WIDTH = 200
INPUT_HEIGHT = 66
INPUT_CHANNELS = 3

VALIDATION_SPLIT = 0.1
USE_REVERSE_IMAGES = False

def customized_loss(y_true, y_pred, loss='euclidean'):
    # Simply a mean squared error that penalizes large joystick summed values
    if loss == 'L2':
        L2_norm_cost = 0.001
        val = K.mean(K.square((y_pred - y_true)), axis=-1) \
            + K.sum(K.square(y_pred), axis=-1) / 2 * L2_norm_cost
    # euclidean distance loss
    elif loss == 'euclidean':
        val = K.sqrt(K.sum(K.square(y_pred - y_true), axis=-1))
    return val


def create_model(keep_prob=0.6):
    
    # CNN for previous frame
    input_previousFrame = Input(shape=(INPUT_HEIGHT, INPUT_WIDTH, INPUT_CHANNELS))
    branch_previousFrame = BatchNormalization()(input_previousFrame)
    branch_previousFrame = Conv2D(24, kernel_size=(5, 5), strides=(2, 2), activation='relu')(branch_previousFrame)
    branch_previousFrame = BatchNormalization()(branch_previousFrame)
    branch_previousFrame = Conv2D(36, kernel_size=(5, 5), strides=(2, 2), activation='relu')(branch_previousFrame)
    branch_previousFrame = BatchNormalization()(branch_previousFrame)
    branch_previousFrame = Conv2D(48, kernel_size=(5, 5), strides=(2, 2), activation='relu')(branch_previousFrame)
    branch_previousFrame = BatchNormalization()(branch_previousFrame)
    branch_previousFrame = Conv2D(64, kernel_size=(3, 3), activation='relu')(branch_previousFrame)
    branch_previousFrame = BatchNormalization()(branch_previousFrame)
    branch_previousFrame = Conv2D(64, kernel_size=(3, 3), activation='relu')(branch_previousFrame)
    branch_previousFrame = Flatten()(branch_previousFrame)
    
    # CNN for current frame
    input_currentFrame = Input(shape=(INPUT_HEIGHT, INPUT_WIDTH, INPUT_CHANNELS))
    branch_currentFrame = BatchNormalization()(input_currentFrame)
    branch_currentFrame = Conv2D(24, kernel_size=(5, 5), strides=(2, 2), activation='relu')(branch_currentFrame)
    branch_currentFrame = BatchNormalization()(branch_currentFrame)
    branch_currentFrame = Conv2D(36, kernel_size=(5, 5), strides=(2, 2), activation='relu')(branch_currentFrame)
    branch_currentFrame = BatchNormalization()(branch_currentFrame)
    branch_currentFrame = Conv2D(48, kernel_size=(5, 5), strides=(2, 2), activation='relu')(branch_currentFrame)
    branch_currentFrame = BatchNormalization()(branch_currentFrame)
    branch_currentFrame = Conv2D(64, kernel_size=(3, 3), activation='relu')(branch_currentFrame)
    branch_currentFrame = BatchNormalization()(branch_currentFrame)
    branch_currentFrame = Conv2D(64, kernel_size=(3, 3), activation='relu')(branch_currentFrame)
    branch_currentFrame = Flatten()(branch_currentFrame)
    
    # Merge CNN outputs     
    concatenated_branches = concatenate([branch_currentFrame, branch_previousFrame])
    concatenated_branches = Dense(1164, activation='relu')(concatenated_branches)
    drop_out = 1 - keep_prob
    concatenated_branches = Dropout(drop_out)(concatenated_branches)
    concatenated_branches = Dense(100, activation='relu')(concatenated_branches)
    concatenated_branches = Dropout(drop_out)(concatenated_branches)
    concatenated_branches = Dense( 50, activation='relu')(concatenated_branches)
    concatenated_branches = Dropout(drop_out)(concatenated_branches)
    concatenated_branches = Dense( 10, activation='relu')(concatenated_branches)
    concatenated_branches = Dropout(drop_out)(concatenated_branches)
    prediction = Dense(OUT_SHAPE, activation='softsign', name="predictions")(concatenated_branches)
    
    model = Model(inputs=[input_previousFrame, input_currentFrame], outputs=prediction)

    return model

def from_bytes(data, big_endian = False):
  if isinstance(data, str):
    data = bytearray(data)
  if big_endian:
    data = reversed(data)
  num = 0
  for offset, byte in enumerate(data):
    num += byte << (offset * 8)
  return num

def is_validation_set(string):
    string_hash = hashlib.md5(string.encode('utf-8')).digest()
    #print string_hash
    #print from_bytes(string_hash[:2], True)
    #print from_bytes(string_hash[:2], True) / 2.0**16
    return from_bytes(string_hash[:2], True) / 2.0**16 > VALIDATION_SPLIT

def load_training_data(track):
    X_train, y_train = [], []
    X_val, y_val = [], []

    if track == 'all':
        recordings = glob.iglob("recordings/*/*/*")
    else:
        recordings = glob.iglob("/home/plodero/neuralcart/recordings/LR/TT/*".format(track))
    idx = 0
    for recording in recordings:
        #print idx
        idx = idx + 1
        filenames = list(glob.iglob('{}/*.png'.format(recording)))
        filenames.sort(key=lambda f: int(os.path.basename(f)[:-4]))

        steering = [float(line) for line in open(
            ("{}/steering.txt").format(recording)).read().splitlines()]

        assert len(filenames) == len(steering), "For recording %s, the number of steering values does not match the number of images." % recording

        for file, steer in zip(filenames, steering):
            assert steer >= -1 and steer <= 1

            valid = is_validation_set(file)
            valid_reversed = is_validation_set(file + '_flipped')

            im = Image.open(file).resize((INPUT_WIDTH, INPUT_HEIGHT))
            im_arr = np.frombuffer(im.tobytes(), dtype=np.uint8)
            im_arr = im_arr.reshape((INPUT_HEIGHT, INPUT_WIDTH, INPUT_CHANNELS))

            if valid:
                X_train.append(im_arr)
                y_train.append(steer)
            else:
                X_val.append(im_arr)
                y_val.append(steer)

            if USE_REVERSE_IMAGES:
                im_reverse = im.transpose(Image.FLIP_LEFT_RIGHT)
                im_reverse_arr = np.frombuffer(im_reverse.tobytes(), dtype=np.uint8)
                im_reverse_arr = im_reverse_arr.reshape((INPUT_HEIGHT, INPUT_WIDTH, INPUT_CHANNELS))

                if valid_reversed:
                    X_train.append(im_reverse_arr)
                    y_train.append(-steer)
                else:
                    X_val.append(im_reverse_arr)
                    y_val.append(-steer)

    assert len(X_train) == len(y_train)
    assert len(X_val) == len(y_val)

    X_train_previous = X_train[-1:] + X_train[:-1]
    X_val_previous = X_val[-1:] + X_val[:-1]
    
    
    return np.asarray(X_train), \
        np.asarray(X_train_previous), \
        np.asarray(y_train).reshape((len(y_train), 1)), \
        np.asarray(X_val), \
        np.asarray(X_val_previous), \
        np.asarray(y_val).reshape((len(y_val), 1))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('track', type=is_valid_track_code)
    parser.add_argument('-c', '--cpu', action='store_true', help='Force Tensorflow to use the CPU.', default=False)
    args = parser.parse_args()

    #if args.cpu:
    #    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    #    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    # Load Training Data
    X_train, X_train_previous, y_train, X_val, X_val_previous, y_val = load_training_data(args.track)

    print(X_train.shape[0], 'training samples.')
    print(X_val.shape[0], 'validation samples.')
    print(X_train_previous.shape[0], 'training samples.')
    print(X_val_previous.shape[0], 'validation samples.')

    # Training loop variables
    epochs = 100
    batch_size = 50

    model = create_model()

    mkdir_p("weights")
    weights_file = "weights/{}.hdf5".format(args.track)
    
    #if os.path.isfile(weights_file):
    #    model.load_weights(weights_file)

    model.compile(loss=customized_loss, optimizer=optimizers.adam(lr=0.0001))
    checkpointer = ModelCheckpoint(
        monitor='val_loss', filepath=weights_file, verbose=1, save_best_only=True, mode='min')
    earlystopping = EarlyStopping(monitor='val_loss', patience=20)
    model.fit([X_train, X_train_previous], y_train, batch_size=batch_size, epochs=epochs,
        shuffle=False, validation_data=([X_val, X_val_previous], y_val), callbacks=[checkpointer, earlystopping])