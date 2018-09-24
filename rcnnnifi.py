# https://gluon-cv.mxnet.io/build/examples_instance/demo_mask_rcnn.html#sphx-glr-build-examples-instance-demo-mask-rcnn-py

from matplotlib import pyplot as plt
from gluoncv import model_zoo, data, utils
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
filename = 'images/rcnn_image_{0}.jpg'.format(uuid)
filename2 = 'images/rcnn_image_p_{0}.jpg'.format(uuid)
cv2.imwrite(filename, frame)


######################################################################
# Load a pretrained model
# -------------------------
#
# Let's get an Mask RCNN model trained on COCO dataset with ResNet-50 backbone.
# By specifying ``pretrained=True``, it will automatically download the model
# from the model zoo if necessary. For more pretrained models, please refer to
# :doc:`../../model_zoo/index`.
#
# The returned model is a HybridBlock :py:class:`gluoncv.model_zoo.MaskRCNN`
# with a default context of `cpu(0)`.

net = model_zoo.get_model('mask_rcnn_resnet50_v1b_coco', pretrained=True)

######################################################################
# Pre-process an image
# --------------------
#
# The pre-processing step is identical to Faster RCNN.
#
# Next we download an image, and pre-process with preset data transforms.
# The default behavior is to resize the short edge of the image to 600px.
# But you can feed an arbitrarily sized image.
#
# You can provide a list of image file names, such as ``[im_fname1, im_fname2,
# ...]`` to :py:func:`gluoncv.data.transforms.presets.rcnn.load_test` if you
# want to load multiple image together.
#
# This function returns two results. The first is a NDArray with shape
# `(batch_size, RGB_channels, height, width)`. It can be fed into the
# model directly. The second one contains the images in numpy format to
# easy to be plotted. Since we only loaded a single image, the first dimension
# of `x` is 1.
#
# Please beware that `orig_img` is resized to short edge 600px.

#im_fname = utils.download('https://github.com/dmlc/web-data/blob/master/' +
#                          'gluoncv/detection/biking.jpg?raw=true',
#                          path='biking.jpg')
x, orig_img = data.transforms.presets.rcnn.load_test(filename)

######################################################################
# Inference and display
# ---------------------
#
# The Mask RCNN model returns predicted class IDs, confidence scores,
# bounding boxes coordinates and segmentation masks.
# Their shape are (batch_size, num_bboxes, 1), (batch_size, num_bboxes, 1)
# (batch_size, num_bboxes, 4), and (batch_size, num_bboxes, mask_size, mask_size)
# respectively. For the model used in this tutorial, mask_size is 14.
#
# Object Detection results
#
# We can use :py:func:`gluoncv.utils.viz.plot_bbox` to visualize the
# results. We slice the results for the first image and feed them into `plot_bbox`:
#
# Plot Segmentation
#
# :py:func:`gluoncv.utils.viz.expand_mask` will resize the segmentation mask
# and fill the bounding box size in the original image.
# :py:func:`gluoncv.utils.viz.plot_mask` will modify an image to
# overlay segmentation masks.

ids, scores, bboxes, masks = [xx[0].asnumpy() for xx in net(x)]

# paint segmentation mask on images directly
width, height = orig_img.shape[1], orig_img.shape[0]
masks = utils.viz.expand_mask(masks, bboxes, (width, height), scores)
orig_img = utils.viz.plot_mask(orig_img, masks)

# identical to Faster RCNN object detection
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(1, 1, 1)
ax = utils.viz.plot_bbox(orig_img, bboxes, scores, ids, class_names=net.classes, ax=ax)
#plt.show()

plt.savefig(filename2)

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