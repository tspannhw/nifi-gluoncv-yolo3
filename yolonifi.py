# See https://gluon-cv.mxnet.io/build/examples_detection/demo_yolo.html#sphx-glr-build-examples-detection-demo-yolo-py
from gluoncv import model_zoo, data, utils
from matplotlib import pyplot as plt
import time
import sys
import datetime
import subprocess
import os
import numpy
import base64
import uuid
import datetime
import traceback
import math
import random, string
import base64
import json
from time import gmtime, strftime
import numpy as np
import cv2
import math
import random, string
import time
import numpy
import random, string
import time
import psutil
import paho.mqtt.client as mqtt
import scipy.misc
from json_tricks import dump, dumps, load, loads, strip_comments
from time import gmtime, strftime
start = time.time()
cap = cv2.VideoCapture(1)   # 0 - laptop   #1 - monitor  #2 external cam
ret, frame = cap.read()
uuid = '{0}_{1}'.format(strftime("%Y%m%d%H%M%S",gmtime()),uuid.uuid4())
filename = 'images/gluoncv_image_{0}.jpg'.format(uuid)
filename2 = 'images/gluoncv_image_p_{0}.jpg'.format(uuid)
cv2.imwrite(filename, frame)

# requires gluoncv 0.3 which is in beta when I did this
# 24-sept-2018
######################################################################
# Load a pretrained model
# -------------------------
#
# Let's get an YOLOv3 model trained with on Pascal VOC
# dataset with Darknet53 as the base model. By specifying
# ``pretrained=True``, it will automatically download the model from the model
# zoo if necessary. For more pretrained models, please refer to
# :doc:`../../model_zoo/index`.

net = model_zoo.get_model('yolo3_darknet53_voc', pretrained=True)

######################################################################
# Pre-process an image
# --------------------
#
# Next we download an image, and pre-process with preset data transforms. Here we
# specify that we resize the short edge of the image to 512 px. You can
# feed an arbitrarily sized image.
# Once constraint for YOLO is that input height and width can be divided by 32.
#
# You can provide a list of image file names, such as ``[im_fname1, im_fname2,
# ...]`` to :py:func:`gluoncv.data.transforms.presets.yolo.load_test` if you
# want to load multiple image together.
#
# This function returns two results. The first is a NDArray with shape
# `(batch_size, RGB_channels, height, width)`. It can be fed into the
# model directly. The second one contains the images in numpy format to
# easy to be plotted. Since we only loaded a single image, the first dimension
# of `x` is 1.

#im_fname = utils.download('https://raw.githubusercontent.com/zhreshold/' +
#                          'mxnet-ssd/master/data/demo/dog.jpg',
#                          path='dog.jpg')
x, img = data.transforms.presets.yolo.load_test(filename, short=512)
#print('Shape of pre-processed image:', x.shape)

######################################################################
# Inference and display
# ---------------------
#
# The forward function will return all detected bounding boxes, and the
# corresponding predicted class IDs and confidence scores. Their shapes are
# `(batch_size, num_bboxes, 1)`, `(batch_size, num_bboxes, 1)`, and
# `(batch_size, num_bboxes, 4)`, respectively.
#
# We can use :py:func:`gluoncv.utils.viz.plot_bbox` to visualize the
# results. We slice the results for the first image and feed them into `plot_bbox`:

class_IDs, scores, bounding_boxs = net(x)

ax = utils.viz.plot_bbox(img, bounding_boxs[0], scores[0], class_IDs[0], class_names=net.classes)
# plt.show()

plt.savefig(filename2)

#print(topn[0][1])
#     top1pct = str(round(topn[0][0],3) * 100)

#print(net.classes[0])
#print(net.classes[1])
#print(scores[0])
#print(class_IDs[0])
#print(bounding_boxs[0])
print(class_IDs[0][0])
print(scores[0][0])
print(bounding_boxs[0][0])
print(net.classes[14])
#print(dumps(net.classes))

end = time.time()
row = { }
row['imgname'] = filename
row['imgnamep'] = filename2
row['host'] = os.uname()[1]
row['shape'] = str(x.shape)
row['end'] = '{0}'.format( str(end ))
row['te'] = '{0}'.format(str(end-start))
row['battery'] = psutil.sensors_battery()[0]
row['systemtime'] = datetime.datetime.now().strftime('%m/%d/%Y %H:%M:%S')
row['cpu'] = psutil.cpu_percent(interval=1)
usage = psutil.disk_usage("/")
row['diskusage'] = "{:.1f} MB".format(float(usage.free) / 1024 / 1024)
row['memory'] = psutil.virtual_memory().percent
row['id'] = str(uuid)
json_string = json.dumps(row)
print(json_string)
