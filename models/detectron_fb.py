from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import gzip
import io
from collections import defaultdict
import argparse
import cv2  # NOQA (Must import before importing caffe2 due to bug in cv2)
import glob
import logging
import os
import sys
import time
from caffe2.python import workspace
import re
import numpy as np


class detectron_fb:
  def __init__(self,time_tracker,model_name='RetinaNet',env_parent='models/',weights_url=None):
    self.tt=time_tracker
    self.model_name = model_name
    self.env_parent = env_parent
    self.env = str(self.env_parent+'env/')
    self.env = self.env
    self.name = 'Detectron'
    self.cocoapi = self.env+'cocoapi'
    self.env_dir = self.env+self.name
    self.pretrained_models_dir = self.env+'pretrained/'
    self.prepared = False
    weights_url = weights_url if weights_url else 'https://dl.fbaipublicfiles.com/detectron/36768744/12_2017_baselines/retinanet_R-101-FPN_1x.yaml.08_31_38.5poQe1ZB/output/train/coco_2014_train%3Acoco_2014_valminusminival/retinanet/model_final.pkl'
    weights_url = str(weights_url)
    self.model = {'weights_url': weights_url}
    
    self.import_file_utils()

    self.model['cfg'] = self.env_dir+'/configs/'+re.search(r'/[0-9]+/(.*yaml)',self.model['weights_url']).groups()[0] #12_2017_baselines/retinanet_X-101-32x8d-FPN_2x.yaml'
    self.model['weights_file']='.'.join(self.model['cfg'].split('/')[-1].split('.')[:-1])+'.pkl'
    self.model['weights']=self.pretrained_models_dir+self.model['weights_file']
    
    
  def download_weights(self):
    download_weights = True
    if os.path.isfile(self.model['weights']) and (os.stat(self.model['weights']).st_size/1048576)>1:
      print('weights file exists. size: ',(os.stat(self.model['weights']).st_size/1048576),'Mb')
      download_weights = False
    if download_weights:
      print('Downloadng weights:')#,self.model['weights_url'],'=>',self.model['weights'])
      self.tt.note_time('Detectron Download weights','begin','dw_weights')
      download_file_urllib(self.model['weights_url'],self.model['weights'])
      self.tt.note_time('Detectron Download weights','end')
      
  def prepare_env(self):
    self.tt.note_time('Detectron Prepare Environment','begin','env_prepare')
    exec_cmd('mkdir -p '+self.env)
    exec_cmd('git clone https://github.com/cocodataset/cocoapi.git '+self.cocoapi)
    exec_cmd('cd '+self.cocoapi+'/PythonAPI && make install')

    exec_cmd('git clone https://github.com/facebookresearch/detectron '+self.env_dir)
    exec_cmd('ln -s '+self.cocoapi+' '+self.env_dir+'cocoapi')
    exec_cmd('pip install -r '+self.env_dir+'/requirements.txt')
    exec_cmd('cd '+self.env_dir+' && make')
    exec_cmd('python '+self.env_dir+'/detectron/tests/test_spatial_narrow_as_op.py')
    exec_cmd('mkdir -p '+self.pretrained_models_dir)
    exec_cmd('ln -s '+self.env_dir+'/detectron detectron')
    self.tt.note_time('Detectron Prepare Environment','end')
    self.download_weights()
      
  def import_dependencies(self):
    global assert_and_infer_cfg,cfg,merge_cfg_from_file,cache_url,setup_logging,Timer,infer_engine,dummy_datasets,c2_utils,vis_utils
    try:
      self.tt.note_time('Detectron import dependecies','begin','import_deps')
      #sys.path.append(str(os.getcwd()+'/'+self.env[:-1])) #remove the trailing /
      detdir = str(os.getcwd()+'/'+self.env_dir)
      if detdir not in sys.path:
        sys.path.append(detdir)
      from detectron.core.config import assert_and_infer_cfg
      from detectron.core.config import cfg
      from detectron.core.config import merge_cfg_from_file
      from detectron.utils.io import cache_url
      from detectron.utils.logging import setup_logging
      from detectron.utils.timer import Timer
      import detectron.core.test_engine as infer_engine
      import detectron.datasets.dummy_datasets as dummy_datasets
      import detectron.utils.c2 as c2_utils
      import detectron.utils.vis as vis_utils
      c2_utils.import_detectron_ops()
      # OpenCL may be enabled by default in OpenCV3; disable it because it's not
      # thread safe and causes unwanted GPU memory allocations.
      cv2.ocl.setUseOpenCL(False)
      
      print('Successfully imported all detectron dependencies')
      return True
    except Exception as e:
      print('Unable to import detectron dependencies\n',e)
    finally:
      self.tt.note_time('Detectron import dependecies','end')
    return False
  
  def prepare(self):
    prepared = self.import_dependencies()
    if not prepared:
      print('Preparing Environment')
      self.prepare_env()
      self.prepared =  self.import_dependencies()
    else:
      self.prepared = True
      print('No need to prepare environment')
    return self.prepared
  
  def load(self):
    if not self.prepared:
      print('NOT PREPARED. QUITTING')
      return
    self.download_weights()
    if not os.path.isfile(self.model['cfg']):
      print(self.model['cfg'],'Not found! . Quitting')
      return
    if not os.path.isfile(self.model['weights']):
      print(self.model['weights'],'Not found! . Quitting')
      return
    
    self.tt.note_time('Detectron load model','begin','load_model')
    merge_cfg_from_file(self.model['cfg'])
    try:
      cfg.NUM_GPUS = 1
      assert_and_infer_cfg(cache_urls=False)
    except Exception as e:
      print('re setting global values may have caused a warning',e)
    
    assert not cfg.MODEL.RPN_ONLY,'RPN models are not supported'
    assert not cfg.TEST.PRECOMPUTED_PROPOSALS,'Models that require precomputed proposals are not supported'

    self.model_obj = infer_engine.initialize_model_from_cfg(self.model['weights'])
    self.dataset = dummy_datasets.get_coco_dataset()
    self.tt.note_time('Detectron load model','end')
    
  def detect_(self,cvimg,opfname='predictions.jpg',visualize=False,threshold=0.7):
    if not self.prepared:
      print('NOT PREPARED. QUITTING')
      return
    
    with c2_utils.NamedCudaScope(0):
      self.tt.interval_start('Detectron detect objects','detect')
      cls_boxes, cls_segms, cls_keyps = infer_engine.im_detect_all(self.model_obj, cvimg, None)
      self.tt.interval_stop('Detectron detect objects')
    
    self.tt.interval_start('Post detection ops','post_detection_ops')
    opdir = '/'.join(opfname.split('/')[:-1])
    opdir = '.' if not opdir else opdir
    opfname = opfname.split('/')[-1]
    opext = opfname.split('.')[-1]
    class_names,bboxes = [],[]
    if visualize:
      opfname = '.'.join(opfname.split('.')[:-1])
      vis_utils.vis_one_image( cvimg[:, :, ::-1],  # BGR -> RGB for visualization
            opfname,
            opdir,
            cls_boxes,
            cls_segms,
            cls_keyps,
            dataset=self.dataset,
            box_alpha=0.3,
            show_class=True,
            thresh=threshold,
            kp_thresh=threshold,
            ext=opext,
            out_when_no_box=True
        )
    else:
      opdir = opdir+'/' if opdir[-1]!='/' else opdir
      cv2.imwrite(opdir+opfname,cvimg)
    boxes, segms, keypoints, class_ids = vis_utils.convert_from_cls_format(cls_boxes, cls_segms, cls_keyps)
    
    # sort in order of largest to smallest order to reduce occlusion
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    sorted_inds = np.argsort(-areas)
    for i in sorted_inds:
      bbox = list(boxes[i, :4])
      score = boxes[i, -1]
      if score < threshold:
          continue
      class_name = str(self.dataset.classes[class_ids[i]])
      class_names.append(class_name)
      bboxes.append(bbox)
        
    return class_names,bboxes

  def detect(self,image,opfile,class_labels_to_filter,visualize=False):
    opfile = opfile if opfile else 'detections.jpg'
    result = self.detect_(image,opfile,visualize=visualize)
    
    if not result:
      result = [],[]
    labels_detected, bboxes = result
    labels_matched = [ str(x) for x in list(set(labels_detected)&set(class_labels_to_filter))]
    print('classes detected:',labels_detected,'classes matched:',labels_matched)
    output_dict = {str('bounding_boxes'):bboxes,str('labels_detected'):labels_detected}
    self.tt.interval_stop('Post detection ops',True if labels_detected else False)
    return {str('labels_matched'): labels_matched, str('output'): output_dict}
  
  def import_file_utils(self):
    '''this function was defined outside class but cython namespace throws error. 
    so this is bad idea i know but i dont want to over complicate this'''
    try:
      sys.path.append('..')
      global exec_cmd,download_file_urllib
      from fileutils import exec_cmd,download_file_urllib
    except Exception as e:
      print('Unable to import fileutils')