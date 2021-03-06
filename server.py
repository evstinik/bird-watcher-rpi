# import the necessary packages
from datetime import datetime
import numpy as np
import imagezmq
import argparse
import imutils
import cv2
import logging
import sys
import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore
import base64
import os

script_folder = os.path.dirname(os.path.realpath(__file__))

def setup_custom_logger(name):
    formatter = logging.Formatter(fmt='%(asctime)s %(levelname)-8s %(message)s',
                                  datefmt='%Y-%m-%d %H:%M:%S')
    #handler = logging.FileHandler('log.txt', mode='w')
    #handler.setFormatter(formatter)
    screen_handler = logging.StreamHandler(stream=sys.stdout)
    screen_handler.setFormatter(formatter)
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    #logger.addHandler(handler)
    logger.addHandler(screen_handler)
    return logger

logger = setup_custom_logger('bird-watcher-server')

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--prototxt", default=os.path.join(script_folder, '../shared-libs/MobileNet-SSD/deploy.prototxt'),
    help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", default=os.path.join(script_folder, '../shared-libs/MobileNet-SSD/mobilenet_iter_73000.caffemodel'),
    help="path to Caffe pre-trained model")
ap.add_argument("-c", "--confidence", type=float, default=0.2,
    help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

logger.info('Connecting to Firebase...')

# initialize firebase
cred = credentials.Certificate(os.path.join(script_folder, 'serviceAccountKey.json'))
firebase_admin.initialize_app(cred)
firebaseDb = firestore.client()
frames_ref = firebaseDb.collection('frames')
stats_ref = firebaseDb.collection('stats')

logger.info('Starting server...')

# initialize the ImageHub object
imageHub = imagezmq.ImageHub()

# initialize the list of class labels MobileNet SSD was trained to
# detect, then generate a set of bounding box colors for each class
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
    "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
    "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
    "sofa", "train", "tvmonitor"]
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

# load our serialized model from disk
logger.info('Loading model...')
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

# load the input image and construct an input blob for the image
# by resizing to a fixed 300x300 pixels and then normalizing it
# (note: normalization is done via the authors of the MobileNet SSD
# implementation)
blob=None
def applySSD(image):
    global blob
    logger.info('Starting recognition...')
    (h, w) = image.shape[0] , image.shape[1]
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 0.007843, (300, 300), 127.5)

    # pass the blob through the network and obtain the detections and
    # predictions
#     print("[INFO] computing object detections...")
    net.setInput(blob)
    detections = net.forward()
    detections_out = []
    
    logger.info('Received result of recognition with total of {} results'.format(detections.shape[2]))

    # loop over the detections
    for i in np.arange(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with the
        # prediction
        confidence = detections[0, 0, i, 2]

        # filter out weak detections by ensuring the `confidence` is
        # greater than the minimum confidence
        if confidence > args["confidence"]:
            # extract the index of the class label from the `detections`,
            # then compute the (x, y)-coordinates of the bounding box for
            # the object
            idx = int(detections[0, 0, i, 1])
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # display the prediction
            label = "{}: {:.2f}%".format(CLASSES[idx], confidence * 100)
            logger.info('#{}: {}'.format(i + 1, label))
            detections_out.append({
                'name': CLASSES[idx],
                'confidence': confidence.item(),
                'bbox': {
                    'startX': startX.item(),
                    'endX': endX.item(),
                    'startY': startY.item(),
                    'endY': endY.item()
                }
            })
            cv2.rectangle(image, (startX, startY), (endX, endY),
                COLORS[idx], 2)
            y = startY - 15 if startY - 15 > 15 else startY + 15
            cv2.putText(image, label, (startX, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)
    return image, detections_out

def deploy(image, detections):
    logger.debug('Deploying to firebase...')
    # Convert to blob
    retval,blob = cv2.imencode('.jpg', image)
    now = datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3] + 'Z'
    # Add entry
    frames_ref.add({
        'image': base64.b64encode(blob),
        'created_at': now,
        'classes': list(map(lambda d: d['name'], detections)),
        'classes_detailed': detections
    })
    # Update stats for today
    today = now[:10]
    if len(detections) > 0:
        stats_ref.document(today).update(
            dict(zip(
                list(map(lambda d: d['name'], detections)),
                [firestore.Increment(1)] * len(detections)
            ))
        )
    logger.debug('Deployed to firebase')

logger.info('Ready')

try:
    # start looping over all the frames
    while True:
        # receive RPi name and frame from the RPi and acknowledge
        # the receipt
        (rpiName, frame) = imageHub.recv_image()
        logger.info('Received frame from {}'.format(rpiName))
        imageHub.send_reply(b'OK')
        # Start recognition
        _,detections = applySSD(frame)
        # Save frame
        #frame_name = '{}_{}.jpg'.format(rpiName, datetime.now().strftime("%y%m%d-%H%M%S.%f"))
        #cv2.imwrite('./frames/{}'.format(frame_name), frame)
        #logger.info("Saved frame to {}".format(frame_name))
        # Deploying original frame only, not with frames
        deploy(frame, detections)
        
except KeyboardInterrupt:
    logger.info("Exiting after interruption")
    
logger.info("Exiting")