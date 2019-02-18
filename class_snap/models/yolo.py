from __future__ import print_function 
from ctypes import *
import math
import os,sys
import cv2
import numpy as np
import random

def import_file_utils():
  try:
    sys.path.append('..')
    global exec_cmd
    from fileutils import exec_cmd,download_file_urllib
  except Exception as e:
    print('Unable to import fileutils')

import_file_utils() # comment this line while running on notebooks

class BOX(Structure):
    _fields_ = [("x", c_float),
                ("y", c_float),
                ("w", c_float),
                ("h", c_float)]


class DETECTION(Structure):
    _fields_ = [("bbox", BOX),
                ("classes", c_int),
                ("prob", POINTER(c_float)),
                ("mask", POINTER(c_float)),
                ("objectness", c_float),
                ("sort_class", c_int)]


class IMAGE(Structure):
    _fields_ = [("w", c_int),
                ("h", c_int),
                ("c", c_int),
                ("data", POINTER(c_float))]


class METADATA(Structure):
    _fields_ = [("classes", c_int),
                ("names", POINTER(c_char_p))]

class yolo:
    def prepare_env(self):
        print('Preparing the environment')
        #if target dir is same as current dir
        if self.td_is_cd:
            exec_cmd('git clone https://github.com/manuhg/darknet '+self.name)
            exec_cmd('mv -v '+self.name+'/* ./')
            exec_cmd('make -j8')
        else:
            exec_cmd('git clone https://github.com/manuhg/darknet '+self.src_dir)
            exec_cmd('cd '+self.src_dir+' && make -j8')
            exec_cmd('cp -v '+self.src_dir+'/libdarknet* '+self.env_dir)
            exec_cmd('ln -s '+self.data_dir+ ' data')
            exec_cmd('ln -s '+self.cfg_dir+ ' cfg')
    
    def load_shared_lib(self,path=None):
        path  = self.shared_lib_path if path is None else path
        if not os.path.isfile(path):
            print(path,' not found')
            return False
        try:
            self.lib = CDLL(path, RTLD_GLOBAL)
            print('Imported libdarknet',self.lib)
            return True
        except Exception as e:
            print('Error loading shared library ',e)
        return False
    
    def c_array(self,ctype, values):
        arr = (ctype*len(values))()
        arr[:] = values
        return arr

    def array_to_image(self,arr):
        arr = arr.transpose(2,0,1)
        c = arr.shape[0]
        h = arr.shape[1]
        w = arr.shape[2]
        arr = (arr/255.0).flatten()
        data = self.c_array(c_float, arr)
        im = IMAGE(w,h,c,data)
        return im
    
    def preprocess_cv_img(self,cvimg):
        cvimg = self.array_to_image(cvimg)
        #self.rgbgr_image(cvimg)
        return cvimg
    
    def convert_to_coordinates(self,result): #once per bbox
        coords = list(map(lambda v: int(v), list(result[2])))
        x,y,w,h = coords #x and y of centre / anchor point
        
        pt1 = x-(w/2),y+(h/2) #left top corner (xmin,ymax)
        pt2 = x+(w/2),y-(h/2) #right bottom cornet
        return pt1,pt2

    def box_and_label(self,result, img,colors): #once per bbox
        #coords = list(map(lambda v: int(v), list(result[2])))
        #x,y,w,h = coords #x and y of centre / anchor point
        
        #pt1 = x-(w/2),y+(h/2) #left top corner (xmin,ymax)
        #pt2 = x+(w/2),y-(h/2) #right bottom cornet
        pt1,pt2 = self.convert_to_coordinates(result)

        label = result[0] + ' - '+str(np.around(float(result[1]), decimals=2))
        color = random.choice(colors)
        cv2.rectangle(img, pt1, pt2, color, 1)
        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1, 1)[0]
        pt2 = pt1[0] + t_size[0] + 3, pt1[1] + t_size[1] + 4
        cv2.rectangle(img, pt1, pt2, color, -1)
        cv2.putText(img, label, (pt1[0], pt1[1] + t_size[1] + 4),cv2.FONT_HERSHEY_PLAIN, 1, [225, 255, 255], 1)
        return img
    
    def detect_(self,net, meta, image, output_file='predictions.jpg', thresh=.5, hier_thresh=.5, nms=.45,visualize=False):
        
        im = self.preprocess_cv_img(image)

        num = c_int(0)
        pnum = pointer(num)
        self.predict_image(net, im)
        dets = self.get_network_boxes(net, im.w, im.h, thresh, hier_thresh, None, 0, pnum)
        num = pnum[0]

        if (nms):
            self.do_nms_obj(dets, num, meta.classes, nms)
        
        res = []
        for j in range(num):
            for i in range(meta.classes):
                if dets[j].prob[i] > 0:
                    b = dets[j].bbox
                    res.append((meta.names[i], dets[j].prob[i], (b.x, b.y, b.w, b.h)))
        res = sorted(res, key=lambda x: -x[1])
        if visualize:
            for r in res:
                image=self.box_and_label(r,image,self.colors)

        cv2.imwrite(output_file,image)
        self.free_detections(dets, num)
        return res
    
    def load(self):
        try:
            print('Files:')
            print(self.model['cfg'],'exists? ',os.path.isfile(self.model['cfg']))
            print(self.model['weights'],'exists?', os.path.isfile(self.model['weights']))
            print(self.labels_data,'exists?',os.path.isfile(self.labels_data))
            
            if not (os.path.isfile(self.model['cfg']) and os.path.isfile(self.model['weights']) and os.path.isfile(self.labels_data)):
              print('Required file not found. Quitting.')
              return
            print('Loading net..',self.model['cfg'], self.model['weights'])
            self.net = self.load_net(self.model['cfg'], self.model['weights'], 0)
            
            print('Loading metadata')
            self.meta = self.load_meta(self.labels_data)
            return True
        except Exception as e:
            print('Error loading network',e)
        return False

    def fusecoordinates(self,coordinates_tuple):
        pt1,pt2 = coordinates_tuple
        return list(pt1)+list(pt2)

    def detect(self,image,opfile,class_labels_to_filter, visualize=False):
        result  = self.detect_(self.net, self.meta, image, opfile)
        print(result)
        labels_detected = [ r[0] for r in result ]
        labels_matched = list(set(labels_detected) & set(class_labels_to_filter))
        bbox_converted =  [ self.fusecoordinates(self.convert_to_coordinates(r)) for r in result  ]
        output_dict = {'bounding_boxes': bbox_converted,'labels_detected':labels_detected}
        return {'labels_matched': labels_matched, 'output': output_dict}

    def __init__(self,model_name='yolov2',prepared = False,env_parent='models/env/',env_dir=None,ptmodels=None):
        self.name='YOLO'
        self.env_dir = 'env_'+self.name+'/' if not env_dir else env_dir
        
        self.pretrained_models_dir = 'pretrained/' if ptmodels is None else ptmodels

        if env_parent:
            self.env_dir = env_parent + self.env_dir
            self.pretrained_models_dir = env_parent + self.pretrained_models_dir

        self.shared_lib_path = self.env_dir+"libdarknet.so"
        self.lib = None
        self.prepared = prepared
        self.model_name = model_name
        self.src_dir = self.env_dir+self.name if env_parent else self.env_dir
        
        self.cfg_dir = self.src_dir+"/cfg/"
        self.labels_data = self.cfg_dir + "coco.data"
        self.data_dir = self.src_dir+"/data"

        self.weights_base_url = 'https://pjreddie.com/media/files/'
        self.cfg_base_url = 'https://raw.githubusercontent.com/pjreddie/darknet/master/'
        self.models_lst = ['yolov2', 'yolov2-tiny', 'yolov3', 'yolov3-tiny']
        self.models = {}
        for model_name_ in self.models_lst:
            self.models.update({model_name_: {'name': model_name_, 'cfg': self.cfg_dir + model_name_+'.cfg', 'weights': self.pretrained_models_dir+model_name_ + '.weights'}})
        self.model_name = model_name
        self.model = self.models[self.model_name]
            
        self.td_is_cd = ''.join(filter(None,self.src_dir.split('/'))) == '.'
        if not self.td_is_cd:
            exec_cmd('mkdir -p '+self.env_dir)
            exec_cmd('mkdir -p '+self.pretrained_models_dir)
        
        self.colors = [(39, 129, 113), (164, 80, 133), (83, 122, 114), (99, 81, 172), (95, 56, 104), (37, 84, 86), (14, 89, 122),
        (80, 7, 65), (10, 102, 25),(90, 185, 109), (106, 110, 132), (169, 158, 85), (188, 185, 26), (103, 1, 17), (82, 144, 81), 
        (92, 7, 184), (49, 81, 155), (179, 177, 69), (93, 187, 158), (13, 39, 73), (12, 50, 60), (16, 179, 33), (112, 69, 165), 
        (15, 139, 63), (33, 191, 159), (182, 173, 32), (34, 113, 133), (90, 135, 34), (53, 34, 86), (141, 35, 190), (6, 171, 8), 
        (118, 76, 112), (89, 60, 55), (15, 54, 88), (112, 75, 181), (42, 147, 38), (138, 52, 63), (128, 65, 149), (106, 103, 24), 
        (168, 33, 45), (28, 136, 135), (86, 91, 108), (52, 11, 76), (142, 6, 189), (57, 81, 168), (55, 19, 148), (182, 101, 89), 
        (44, 65, 179), (1, 33, 26), (122, 164, 26), (70, 63, 134), (137, 106, 82), (120, 118, 52), (129, 74, 42), (182, 147, 112),
        (22, 157, 50), (56, 50, 20), (2, 22, 177), (156, 100, 106), (21, 35, 42), (13, 8, 121), (142, 92, 28), (45, 118, 33), 
        (105, 118, 30), (7, 185, 124), (46, 34, 146), (105, 184, 169), (22, 18, 5), (147, 71, 73), (181, 64, 91), (31, 39, 184),
        (164, 179, 33), (96, 50, 18), (95, 15, 106), (113, 68, 54), (136, 116, 112), (119, 139, 130), (31, 139, 34), (66, 6, 127),
        (62, 39, 2), (49, 99, 180), (49, 119, 155), (153, 50, 183), (125, 38, 3), (129, 87, 143), (49, 87, 40), (128, 62, 120),
        (73, 85, 148), (28, 144, 118), (29, 9, 24), (175, 45, 108), (81, 175, 64), (178, 19, 157), (74, 188, 190), (18, 114, 2),
        (62, 128, 96), (21, 3, 150), (0, 6, 95), (2, 20, 184), (122, 37, 185)]
    
    def print_config(self):
        print('env_dir:',self.env_dir,'\nsrc_dir',self.src_dir,'\npretrained models dir',self.pretrained_models_dir,'\ncfg dir',self.cfg_dir,'\ndata dir',self.data_dir)

    def prepare(self,prepared=None):        
        prepared = self.load_shared_lib()
        if not prepared:
            print('Need to prepare environment')
            self.prepare_env()
            prepared = self.load_shared_lib()
            if not prepared:
              print('ERROR LOADING libdarknet.so')
              return
        
        download_wights = True
        if os.path.isfile(self.model['weights']) and (os.stat('yolov3-tiny.weights').st_size/1048576)>1:
          print('weights file exists. size: ',(os.stat('yolov3-tiny.weights').st_size/1048576),'Mb')
          download_wights = False
                
        if download_wights:
            target_file = self.model_name+'.weights'
            if not self.td_is_cd:
                target_file = self.pretrained_models_dir+self.model_name+'.weights'
            download_file_urllib('https://pjreddie.com/media/files/'+self.model_name+'.weights',target_file)
        
        lib = self.lib
        
        lib.network_width.argtypes = [c_void_p]
        lib.network_width.restype = c_int
        lib.network_height.argtypes = [c_void_p]
        lib.network_height.restype = c_int

        #load_alphabet,draw_detections,save_image,    letterbox_image
        load_alphabet = lib.load_alphabet
        load_alphabet.argtypes = []
        load_alphabet.restype = POINTER(POINTER(IMAGE))
        
        # self.draw_detections = lib.draw_detections
        # self.draw_detections.argtypes = [IMAGE, POINTER(DETECTION), c_int, c_float, POINTER(c_char_p), POINTER(POINTER(IMAGE)), c_int]
        # self.draw_detections.restype = IMAGE
        
        self.save_image = lib.save_image
        self.save_image.argtypes = [IMAGE, c_char_p]
        
        self.predict = lib.network_predict
        self.predict.argtypes = [c_void_p, POINTER(c_float)]
        self.predict.restype = POINTER(c_float)
        
        self.set_gpu = lib.cuda_set_device
        self.set_gpu.argtypes = [c_int]
        
        self.make_image = lib.make_image
        self.make_image.argtypes = [c_int, c_int, c_int]
        self.make_image.restype = IMAGE
        
        self.get_network_boxes = lib.get_network_boxes
        self.get_network_boxes.argtypes = [c_void_p, c_int, c_int,c_float, c_float, POINTER(c_int), c_int, POINTER(c_int)]
        self.get_network_boxes.restype = POINTER(DETECTION)

        self.make_network_boxes = lib.make_network_boxes
        self.make_network_boxes.argtypes = [c_void_p]
        self.make_network_boxes.restype = POINTER(DETECTION)

        self.free_detections = lib.free_detections
        self.free_detections.argtypes = [POINTER(DETECTION), c_int]

        self.free_ptrs = lib.free_ptrs
        self.free_ptrs.argtypes = [POINTER(c_void_p), c_int]

        self.network_predict = lib.network_predict
        self.network_predict.argtypes = [c_void_p, POINTER(c_float)]

        self.reset_rnn = lib.reset_rnn
        self.reset_rnn.argtypes = [c_void_p]

        self.load_net = lib.load_network
        self.load_net.argtypes = [c_char_p, c_char_p, c_int]
        self.load_net.restype = c_void_p

        self.do_nms_obj = lib.do_nms_obj
        self.do_nms_obj.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]

        self.do_nms_sort = lib.do_nms_sort
        self.do_nms_sort.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]

        self.free_image = lib.free_image
        self.free_image.argtypes = [IMAGE]

        # self.letterbox_image = lib.letterbox_image
        # self.letterbox_image.argtypes = [IMAGE, c_int, c_int]
        # self.letterbox_image.restype = IMAGE

        self.load_meta = lib.get_metadata
        self.lib.get_metadata.argtypes = [c_char_p]
        self.lib.get_metadata.restype = METADATA

        self.load_image = lib.load_image_color
        self.load_image.argtypes = [c_char_p, c_int, c_int]
        self.load_image.restype = IMAGE

        self.rgbgr_image = lib.rgbgr_image
        self.rgbgr_image.argtypes = [IMAGE]

        self.predict_image = lib.network_predict_image
        self.predict_image.argtypes = [c_void_p, IMAGE]
        self.predict_image.restype = POINTER(c_float)
    


  
# if __name__=="__main__":
#     y = yolo(model_name='yolov3-tiny')#,env_parent=None,env_dir='./',ptmodels='./')
#     y.prepare()
#     y.load()