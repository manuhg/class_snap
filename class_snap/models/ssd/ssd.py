import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
from distutils.version import StrictVersion
from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image
import cv2,time

def exec_cmd(cms_str,echo=True):
    print(os.popen(cmd_str).read() if echo else '',end='')

def prepare_env():
    #env preparation since this notebook is being run as standalone
    if StrictVersion(tf.__version__.split('-')[0]) < StrictVersion('1.9'):
        raise ImportError('Please upgrade your TensorFlow installation to v1.9.* or later!')

    print('Using OpenCV version %r and Tensorflow version %r'%(cv2.__version__,tf.__version__))
    # exec_cmd('rm -rf *'))
    # exec_cmd('git clone https://github.com/tensorflow/models.git md --recursive'))
    # exec_cmd('git clone https://github.com/cocodataset/cocoapi.git'))
    # exec_cmd('cd cocoapi/PythonAPI && make && cp -rv pycocotools ../../md/research/'))
    # exec_cmd('cd md/research && protoc object_detection/protos/*.proto --python_out=.'))
    # exec_cmd('cd md/research && python setup.py install'))
    # exec_cmd('cd md/research/slim && python setup.py install'))
    # exec_cmd('cd md/research && python object_detection/builders/model_builder_test.py'))
    # exec_cmd('mv -v md/research/* ./'))
    # #mv md/research/object_detection ./'
    # exec_cmd('mv -v md/research/setup.py ./'))
    # exec_cmd('rm -rf md'))
    # exec_cmd('ls'))
    # exec_cmd('python object_detection/builders/model_builder_test.py'))

def prerequisits():
    # This is needed since the notebook is stored in the object_detection folder.
    sys.path.append("..")
    sys.path.append("object_detection")
    from object_detection.utils import ops as utils_ops
    from object_detection import utils
    from utils import label_map_util
    from utils import visualization_utils as vis_util
    !cp -rv object_detection/* ./