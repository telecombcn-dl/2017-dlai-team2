import sys, time, logging, os, argparse

import numpy as np
from PIL import Image, ImageGrab, ImageMath, ImageChops
from socketserver import TCPServer, StreamRequestHandler

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
#config.gpu_options.per_process_gpu_memory_fraction = 0.3
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from train_new_map import extract_map, create_model, is_valid_track_code, INPUT_WIDTH, INPUT_HEIGHT, INPUT_CHANNELS, XMIN, XMAX, YMIN, YMAX, MAP_HEIGHT, MAP_WIDTH, INPUT_CHANNELS_MAP

def prepare_image(im):
    im = im.resize((INPUT_WIDTH, INPUT_HEIGHT))
    im_arr = np.frombuffer(im.tobytes(), dtype=np.uint8)
    im_arr = im_arr.reshape((INPUT_HEIGHT, INPUT_WIDTH, INPUT_CHANNELS))
    im_arr = np.expand_dims(im_arr, axis=0)
    return im_arr

class TCPHandler(StreamRequestHandler):
    def handle(self):
        if args.all:
            weights_file = 'weights/all.hdf5'
            logger.info("Loading {}...".format(weights_file))
            model.load_weights(weights_file)

        logger.info("Handling a new connection...")
        for line in self.rfile:
            message = str(line.strip(),'utf-8')
            logger.info(message)

            if message.startswith("COURSE:") and not args.all:
                course = message[7:].strip().lower()
                weights_file = 'weights/{}.hdf5'.format(course)
                logger.info("Loading {}...".format(weights_file))
                model.load_weights(weights_file)

            if message.startswith("PREDICTFROMCLIPBOARD"):
                logger.info("Predicting from clipboard")
                im = ImageGrab.grabclipboard()
                if im != None:
                    # insert extraction here
                    prediction = model.predict([prepare_image(im), np.expand_dims(extract_map(im, Image.open("masks/mm_LR.png").convert("RGB")), axis=0)], batch_size=1)[0]
                    self.wfile.write((str(prediction[0]) + "\n").encode('utf-8'))
                    logger.info("Im != null")
                else:
                    self.wfile.write("PREDICTIONERROR\n".encode('utf-8'))
                    logger.info("Im == null --> Prediction Error!")

            if message.startswith("PREDICT:"):
                logger.info("Predict")
                im = Image.open(message[8:])
                logger.info(message)
                prediction = model.predict([prepare_image(im), np.expand_dims(extract_map(im, Image.open("masks/mm_LR.png").convert("RGB")), axis=0)], batch_size=1)[0]
                self.wfile.write((str(prediction[0]) + "\n").encode('utf-8'))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Start a prediction server that other apps will call into.')
    parser.add_argument('-a', '--all', action='store_true', help='Use the combined weights for all tracks, rather than selecting the weights file based off of the course code sent by the Play.lua script.', default=False)
    parser.add_argument('-p', '--port', type=int, help='Port number', default=36296)
    parser.add_argument('-c', '--cpu', action='store_true', help='Force Tensorflow to use the CPU.', default=False)
    args = parser.parse_args()

    if args.cpu:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    logger.info("Loading model...")
    model = create_model(keep_prob=1)

    if args.all:
        model.load_weights('weights/all.hdf5')

    logger.info("Starting server...")
    server = TCPServer(('0.0.0.0', args.port), TCPHandler)

    print("Listening on Port: {}".format(server.server_address[1]))
    sys.stdout.flush()
    server.serve_forever()
