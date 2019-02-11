from fileutils import *
from PIL import Image
from matplotlib import pyplot as plt
from io import StringIO
from collections import defaultdict
from distutils.version import StrictVersion
import cv2
import time
import zipfile
import tensorflow as tf
import tarfile
import sys
import six.moves.urllib as urllib
import os
import numpy as np


class ssd:
    def __init__(self, prepared=False):
        self.name = 'SSD'
        self.prepared = prepared
        self.base_url = 'http://download.tensorflow.org/models/object_detection/'
        self.available_models = ['ssd_mobilenet_v1_coco_2017_11_17',
                                 'ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03', 'faster_rcnn_nas_coco_2018_01_28']
        self.model_name = self.available_models[-1]
        self.model_file = self.model_name + '.tar.gz'
        self.frozen_graph_name = 'frozen_inference_graph.pb'
        self.frozen_graph = self.model_name + '/' + self.frozen_graph_name

        if StrictVersion(tf.__version__.split('-')[0]) < StrictVersion('1.9'):
            raise ImportError(
                'Please upgrade your TensorFlow installation to v1.9.* or later!')
        print('Using OpenCV version %r and Tensorflow version %r' %
              (cv2.__version__, tf.__version__))

    def prepare(self, prepared=None):
        prepared = self.prepared if prepared is None else prepared
        if not prepared:
            self.prepare_env()
        else:
            print('No need to prepare environment')
        self.prepared = self.import_utils()
        self.lalbels_file = os.path.join('data', 'mscoco_label_map.pbtxt')
        download_file(self.base_url + self.model_file, self.model_file)
        extract_file_from_tar(self.model_file, self.frozen_graph_name)
        self.detection_graph = get_detection_graph(self.frozen_graph)
        self.category_index = label_map_util.create_category_index_from_labelmap(
            self.lalbels_file, use_display_name=True)

    def detect(self, image, opfile, class_labels_to_filter, detection_graph=None, category_index=None):
        print(' '+opfile.split('/')[-1], end=' ')
        detection_graph = self.detection_graph if not detection_graph else detection_graph
        category_index = self.category_index if not category_index else category_index
        image_np = image
        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
        image_np_expanded = np.expand_dims(image_np, axis=0)
        # Actual detection.
        output_dict = self.run_inference_for_single_image(
            image_np, detection_graph)
        # Visualization of the results of a detection.
        class_labels_detected = [category_index[obj]['name']
                                 for obj in output_dict['detection_classes']]
        labels_matched = set(class_labels_detected) & set(
            class_labels_to_filter)
        print(labels_matched)
        vis_util.visualize_boxes_and_labels_on_image_array(
            image_np, output_dict['detection_boxes'], output_dict['detection_classes'], output_dict['detection_scores'],
            category_index, instance_masks=output_dict.get('detection_masks'), use_normalized_coordinates=True, line_thickness=8)
        cv2.imwrite(opfile, image_np)
        return labels_matched

    def prepare_env(self):
        print('preparing environment')
        # env preparation since this notebook is being run as standalone
        exec_cmd('rm -rf * ')
        exec_cmd('git clone https://github.com/tensorflow/models.git md --recursive')
        exec_cmd('git clone https://github.com/cocodataset/cocoapi.git')
        exec_cmd(
            'cd cocoapi/PythonAPI && make && cp -rv pycocotools ../../md/research/')
        exec_cmd(
            'cd md/research && protoc object_detection/protos/*.proto --python_out=.')
        exec_cmd('cd md/research && python setup.py install')
        exec_cmd('cd md/research/slim && python setup.py install')
        exec_cmd(
            'cd md/research && python object_detection/builders/model_builder_test.py')
        exec_cmd('mv -v md/research/* ./')
        #exec_cmd('mv md/research/object_detection ./')
        exec_cmd('mv -v md/research/setup.py ./')
        exec_cmd('rm -rf md')
        exec_cmd('ls')
        exec_cmd('python object_detection/builders/model_builder_test.py')

    def import_utils(self):
        try:
            sys.path.append("..")
            sys.path.append("object_detection")
            global utils_ops, utils, label_map_util, visualization_utils, vis_util
            from object_detection.utils import ops as utils_ops
            from object_detection import utils
            from utils import label_map_util
            from utils import visualization_utils as vis_util

            if not self.prepared:
                exec_cmd('cp -rv object_detection/* ./')
        except Exception as e:
            print('\n\nError preparing environment : ', e)
            return False
        return True

    def run_inference_for_single_image(self, image, graph):
        with graph.as_default():
            with tf.Session() as sess:
                # Get handles to input and output tensors
                ops = tf.get_default_graph().get_operations()
                all_tensor_names = {
                    output.name for op in ops for output in op.outputs}
                tensor_dict = {}
                for key in ['num_detections', 'detection_boxes', 'detection_scores', 'detection_classes', 'detection_masks']:
                    tensor_name = key + ':0'
                    if tensor_name in all_tensor_names:
                        tensor_dict[key] = tf.get_default_graph(
                        ).get_tensor_by_name(tensor_name)
                if 'detection_masks' in tensor_dict:
                    # The following processing is only for single image
                    detection_boxes = tf.squeeze(
                        tensor_dict['detection_boxes'], [0])
                    detection_masks = tf.squeeze(
                        tensor_dict['detection_masks'], [0])
                    # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
                    real_num_detection = tf.cast(
                        tensor_dict['num_detections'][0], tf.int32)
                    detection_boxes = tf.slice(detection_boxes, [0, 0], [
                                               real_num_detection, -1])
                    detection_masks = tf.slice(detection_masks, [0, 0, 0], [
                                               real_num_detection, -1, -1])
                    detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
                        detection_masks, detection_boxes, image.shape[0], image.shape[1])
                    detection_masks_reframed = tf.cast(
                        tf.greater(detection_masks_reframed, 0.5), tf.uint8)
                    # Follow the convention by adding back the batch dimension
                    tensor_dict['detection_masks'] = tf.expand_dims(
                        detection_masks_reframed, 0)
                image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')
                # Run inference
                output_dict = sess.run(tensor_dict, feed_dict={
                                       image_tensor: np.expand_dims(image, 0)})
                # all outputs are float32 numpy arrays, so convert types as appropriate
                output_dict['num_detections'] = int(
                    output_dict['num_detections'][0])
                output_dict['detection_classes'] = output_dict['detection_classes'][0].astype(
                    np.uint8)
                output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
                output_dict['detection_scores'] = output_dict['detection_scores'][0]
                if 'detection_masks' in output_dict:
                    output_dict['detection_masks'] = output_dict['detection_masks'][0]
                return output_dict

    def get_detection_graph(self, frozen_graph):
        detection_graph = tf.Graph()
        with detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(frozen_graph, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')
        return detection_graph
