import numpy as np
import cv2
import time

start = time.process_time()

confidence_thr = 0.5

CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
    "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
    "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
    "sofa", "train", "tvmonitor"]
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

print("[INFO] loading model...")

shared_dir = '../shared-libs/MobileNet-SSD/'
net = cv2.dnn.readNetFromCaffe(shared_dir + 'deploy.prototxt', shared_dir + 'mobilenet_iter_73000.caffemodel')

# load the input image and construct an input blob for the image
# by resizing to a fixed 300x300 pixels and then normalizing it
# (note: normalization is done via the authors of the MobileNet SSD
# implementation)
blob=None
def applySSD(image):
    global blob
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 0.007843, (300, 300), 127.5)

    # pass the blob through the network and obtain the detections and
    # predictions
#     print("[INFO] computing object detections...")
    net.setInput(blob)
    detections = net.forward()

    # loop over the detections
    for i in np.arange(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with the
        # prediction
        confidence = detections[0, 0, i, 2]

        # filter out weak detections by ensuring the `confidence` is
        # greater than the minimum confidence
        if confidence > confidence_thr:
            # extract the index of the class label from the `detections`,
            # then compute the (x, y)-coordinates of the bounding box for
            # the object
            idx = int(detections[0, 0, i, 1])
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # display the prediction
            label = "{}: {:.2f}%".format(CLASSES[idx], confidence * 100)
            print("[INFO] {}".format(label))
            cv2.rectangle(image, (startX, startY), (endX, endY),
                COLORS[idx], 2)
            y = startY - 15 if startY - 15 > 15 else startY + 15
            cv2.putText(image, label, (startX, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)
    return image

end_loading = time.process_time()
print("[INFO] Model loaded in {:.2f}".format(end_loading - start))

# create input blob
frame = cv2.imread('./examples/bird-from-balkony.jpeg')
end_image = time.process_time()
print("[INFO] Image loaded in {:.2f}".format(end_image - end_loading))

(h, w) = frame.shape[0] , frame.shape[1]
frame = applySSD(frame)
end_recognition = time.process_time()
print("[INFO] Recognition finished in {:.2f}".format(end_recognition - end_image))

cv2.imwrite('./output.jpg', frame)
end_output = time.process_time()
print("[INFO] Image written in {:.2f}".format(end_output - end_recognition))

end = time.process_time()
print("[INFO] Total {:.2f}".format(end - start))