from __future__ import print_function
import cv2
import json
import os,sys,time
from fileutils import download_file, exec_cmd,save_as_annotations
from detector import detector
import shutil

class extractor:
    def __init__(self,model_name='yolo',model_variant=None,load=True):
        self.model_name = model_name
        self.model_variant = model_variant
        self.prepare()
        self.load()
    
    def prepare(self,model_name=None,model_variant=None):
        model_name = model_name if model_name else self.model_name
        model_name = model_name if model_name else 'yolo'
        self.model_name = model_name
        
        model_variant = model_variant if model_variant else self.model_variant
        model_variant = model_variant if model_variant else 'yolov2'
        self.model_variant = model_variant

        self.detector_model = detector(model_name,model_variant=model_variant).get_model()
        self.detector_model.prepare()
    
    def load(self):
        self.detector_model.load()

    def process(self,input_file,class_labels_to_filter_by,interval=1,dest_dir='output',zip_name='detections.zip',del_after=True,visualize=False):
        #process
        self.class_labels_to_filter_by = class_labels_to_filter_by
        self.interval = interval
        dest_dir = dest_dir if dest_dir else 'output'
        self.dest_dir = dest_dir

        #detection / extraction
        exec_cmd('mkdir '+dest_dir)
        output = self.extract_frames(self.detector_model, input_file, class_labels_to_filter_by, interval=interval,dest_dir=dest_dir)
        self.output = output

        #annotation
        json_files,failures,annotated_output = save_as_annotations(output,dest_dir)
        self.annotated_output = annotated_output
        if failures:
            print('Failed to write: ',','.join(failures))

        #moving and re organisation as data and meta
        opdir = dest_dir+'/detections/'
        data_dir = opdir+'/data'
        meta_dir = opdir+'/meta'
        exec_cmd('rm -rf '+opdir)
        exec_cmd('mkdir -p '+data_dir)
        exec_cmd('mkdir -p '+meta_dir)
        
        #json_files = [ "'"+jf+"'" for jf in json_files]
        #jpg_files = ["'"+dest_dir+'/'+f+"'" for f in list(output.keys())]
        jpg_files = [dest_dir+'/'+f for f in list(output.keys())]
        print(json_files,jpg_files)
        exec_cmd('cp '+' '.join(jpg_files)+' '+data_dir)
        exec_cmd('cp '+' '.join(json_files)+' '+meta_dir)
        #create zip
        #exec_cmd('zip '+zip_name+' -r '+opdir)
        if zip_name.endswith('zip'):
            zip_name = zip_name[:zip_name.rfind('.')]
        shutil.make_archive(zip_name, 'zip', opdir)


        if del_after:
            exec_cmd('rm -rf '+dest_dir)
        return annotated_output
    
    def extract_frames(self,detector, input_file, class_labels, interval=None, dest_dir='.',visualize=False):# interval if specified should be in terms of seconds
        print('Detection Algorithm:%s\nModel loaded: %s\nInput File: %s\nIntervals at which to detect: %r seconds' % (detector.name, detector.model_name, input_file, interval))
        interval = interval * 1000  # convert to milliseconds
        target_interval = interval
        cap = cv2.VideoCapture(input_file)
        output = {}  # format: File name : [list of matched labels]
        if (cap.isOpened() == False):
            print("Error opening video file", input_file)
        i = 0
        fps = cap.get(cv2.CAP_PROP_FPS)
        input_file_name = '.'.join(input_file.split('.')[:-1])
        total_duration = 0
        input_file_name = input_file_name.replace("\\ ", " ")

        if dest_dir[-1] != '/':
            dest_dir = dest_dir + '/'

        while(cap.isOpened()):
            opfname = None
            if interval:
                cap.set(cv2.CAP_PROP_POS_MSEC, target_interval)
                opfname = input_file_name+'-'+str(int(target_interval/1000)).rjust(4,'0')+'s.jpg'
                target_interval += interval
            ret, frame = cap.read()
            
            if not ret:
                break
            
            i += 1
            print('Frame: ', i, end=' ')
            #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            opfname = opfname if opfname else input_file_name + '-'+str(int(i/fps))+':'+str(i % fps)+'.jpg'
            
            t1 = time.time()
            result = detector.detect(frame, dest_dir+opfname, class_labels,visualize=visualize)
            t2 = time.time()
            duration = t2-t1

            total_duration += duration
            if result and result['labels_matched']:
                result.update({'time': duration})
                output.update({opfname: result})
            else:
                print(' - No labels matched. Frame rejected')
            
        print('Frames processed :', i)
        print('Overall Processing speed per image', (total_duration)/i)
        print('Total duration:',total_duration)
        cap.release()
        cv2.destroyAllWindows()
        return output
