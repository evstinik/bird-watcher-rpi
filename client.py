# import the necessary packages
from imutils.video import VideoStream
import imutils
import imagezmq
import argparse
import socket
import time
import cv2
import logging
import sys

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

logger = setup_custom_logger('bird-watcher-client')

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-s", "--server-ip", required=True,
    help="ip address of the server to which the client will connect")
ap.add_argument("-a", "--min-area", type=int, default=500, help="minimum area size")
args = vars(ap.parse_args())

logger.info('Starting client...')

# initialize the ImageSender object with the socket address of the
# server
sender = imagezmq.ImageSender(connect_to="tcp://{}:5555".format(
    args["server_ip"]))

logger.info('Connected to server')

# get the host name, initialize the video stream, and allow the
# camera sensor to warmup
rpiName = socket.gethostname()
vs = VideoStream(usePiCamera=True, resolution=(320, 240)).start()
time.sleep(2.0)

logger.info('Camera warmed up')
 
avg_frame = None
def check_motion(frame):
    global avg_frame
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)
    # if the first frame is None, initialize it
    if avg_frame is None:
        avg_frame = gray.copy().astype(float)
        return False
    
    # compute the absolute difference between the current frame and
    # first frame
    cv2.accumulateWeighted(gray, avg_frame, 0.5)
    frameDelta = cv2.absdiff(gray, cv2.convertScaleAbs(avg_frame))
    thresh = cv2.threshold(frameDelta, 25, 255, cv2.THRESH_BINARY)[1]
    # dilate the thresholded image to fill in holes, then find contours
    # on thresholded image
    thresh = cv2.dilate(thresh, None, iterations=2)
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    
    avg_frame = gray.copy().astype(float)
    
    if len(cnts) > 0:
        logger.debug('Detected {} contours'.format(len(cnts)))
    
    # loop over the contours
    for c in cnts:
        # if the contour is too small, ignore it
        area = cv2.contourArea(c)
        if area < args["min_area"]:
            logger.debug('Contour has too small area {}'.format(area))
            continue
        logger.debug('Contour has big enough area {}'.format(area))
        return True
    
    return False
    

try:
    while True:
        # read the frame from the camera and send it to the server
        frame = vs.read()
        
        if check_motion(frame):
            logger.info('Detected motion')
            sender.send_image(rpiName, frame)
            logger.debug('Frame send')
            # Do not track too many photos
            time.sleep(2)
            
        time.sleep(1)
except KeyboardInterrupt:
    vs.stop()
    
vs.stop()