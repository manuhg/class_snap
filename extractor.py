from __future__ import print_function
import cv2
import json
import os,sys,time
from fileutils import download_file,save_dicts_to_file, exec_cmd,save_as_annotations,download_youtube_video,import_pytube

from detector import detector
import shutil

from time_tracker import time_tracker as tcc

class extractor:
    def __init__(self,model_name='yolo',model_variant=None,load=True):
        self.name = model_name + '' if model_variant is None else '-'+model_variant
        self.tc = tcc(self.name,target_type='extractor')
        self.tc.note_time('Total Time','begin','Total_time')
        self.model_name = model_name
        self.model_variant = model_variant
        self.prepare()
        self.load()
    
    def prepare(self,model_name=None,model_variant=None):
        self.tc.note_time('Init & Prepare model','begin','init&prepare_model')
        model_name = model_name if model_name else self.model_name
        model_name = model_name if model_name else 'yolo'
        self.model_name = model_name
        
        model_variant = model_variant if model_variant else self.model_variant
        model_variant = model_variant if model_variant else 'yolov2'
        self.model_variant = model_variant

        self.detector_model = detector(model_name,model_variant=model_variant,time_tracker=tcc(name=self.name,target_type='detection_model')).get_model()
        self.detector_model.prepare()
        self.tc.note_time('Init & Prepare model','end')
    
    def load(self):
        self.tc.note_time('Load Object Detection Model','begin','load_model')
        self.detector_model.load()
        import_pytube()
        self.tc.note_time('Load Object Detection Model','end')
        

    def process(self,input_file,class_labels_to_filter_by,interval=1,dest_dir='output',zip_name='detections.zip',del_after=True,visualize=False):
        #################### pre process ####################
        tmpdir = 'tmp'
        self.class_labels_to_filter_by = class_labels_to_filter_by
        self.interval = interval
        dest_dir = dest_dir if dest_dir else 'output'
        self.dest_dir = dest_dir
        if not os.path.isfile(input_file):
            self.tc.note_time('Download video from Youtube','begin','download_video')
            filename = download_youtube_video(input_file)
            if filename and os.path.isfile(filename):
                filename = filename.replace("\\ ", " ")
                input_file = filename
            else:
                print('Not a valid yotube video url or unable to download video from url')
                return None,None
            self.tc.note_time('Download video from Youtube','end')
        exec_cmd('mkdir '+tmpdir)
        
        #################### detection / extraction ####################
        output,total_duration,successful = self.extract_frames(self.detector_model, input_file, class_labels_to_filter_by, interval=interval,dest_dir=tmpdir,visualize=visualize)
        save_dicts_to_file(output,'output_data.json')
        self.output = output
        if successful<1:
            print('NO OBJECTS SPECIFIED WERE DETECTED!. Hence no annotations to be saved')
            exit()
        
        #################### annotation ####################
        self.tc.note_time('Save output as annotations','begin','save_annotations')
        json_files,failures,annotated_output = save_as_annotations(output,tmpdir)
        self.annotated_output = annotated_output
        if failures:
            print('Failed to write: ',','.join(failures))
        ###moving and re organisation as data and meta
        opdir = tmpdir+'/detections/'
        data_dir = opdir+'/data'
        meta_dir = opdir+'/meta'
        exec_cmd('rm -rf '+opdir)
        exec_cmd('mkdir -p '+data_dir)
        exec_cmd('mkdir -p '+meta_dir)
        
        json_files = [ "'"+jf+"'" for jf in json_files]
        jpg_files = ["'"+tmpdir+'/'+f+"'" for f in list(output.keys())]

        exec_cmd('cp '+' '.join(jpg_files)+' '+data_dir)
        exec_cmd('cp '+' '.join(json_files)+' '+meta_dir)
        #create zip
        #exec_cmd('zip '+zip_name+' -r '+opdir)
        if zip_name.endswith('zip'):
            zip_name = zip_name[:zip_name.rfind('.')]
        shutil.make_archive(zip_name, 'zip', opdir)
        zip_name +='.zip'


        if del_after:
            exec_cmd('rm -rf '+tmpdir)

        if dest_dir != '.' and dest_dir!='./':
            exec_cmd('mkdir -p '+dest_dir)
            exec_cmd("mv '"+zip_name+"' "+dest_dir)
        
        self.add_data_tts({'input_file':input_file})

        self.tc.note_time('Save output as annotations','end')
        self.tc.note_time('Total Time','end')
        print('\n###############################\n')
        print('Overall Time Taken summary')
        self.tc.summary()
        
        print('\n###############################\n')
        print('Detector model time taken summary')
        self.detector_model.tt.summary()
        print('\n')
        save_dicts_to_file(self.detector_model.tt.intervals,'det-intervals_data.json')
        save_dicts_to_file(self.tc.intervals,'ext-intervals_data.json')
        return annotated_output,total_duration
    
    def add_data_tts(self,dct):
        self.detector_model.tt.add_data(dct)
        self.tc.add_data(dct)

    def extract_frames(self,detector, input_file, class_labels, interval=None, dest_dir='.',visualize=False):# interval if specified should be in terms of seconds
        self.tc.note_time('Load video file for frame extraction','begin','load_video_file')
        print('Detection Algorithm:%s\nInput File: %s\nIntervals at which to detect: %r seconds' % (detector.name, input_file, interval))
        interval = interval * 1000  # convert to milliseconds
        target_interval = interval
        cap = cv2.VideoCapture(input_file)
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_duration = frameCount/fps
        self.add_data_tts({'video_duration':video_duration})

        file_size = round(os.stat(input_file).st_size/1048576) #convert to Mbs
        self.add_data_tts({'video_file_size':file_size})

        output = {}  # format: File name : [list of matched labels]
        if (cap.isOpened() == False):
            print("Error opening video file", input_file)
        i = 0

        successful = 0
        fps = cap.get(cv2.CAP_PROP_FPS)
        input_file_name = '.'.join(input_file.split('.')[:-1])
        total_duration = 0
        input_file_name = input_file_name.replace("\\ ", " ")

        if dest_dir[-1] != '/':
            dest_dir = dest_dir + '/'

        self.tc.note_time('Load video file for frame extraction','end')
        
        while(cap.isOpened()):
            self.tc.interval_start('Fetch frame','fetch_frame')
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
            opfname = opfname.split('/')[-1]
            self.tc.interval_stop('Fetch frame')
            self.tc.interval_start('Detect objects in frame','detect')
            
            t1 = time.time()
            result = detector.detect(frame, dest_dir+opfname, class_labels,visualize=visualize)
            duration = time.time()-t1

            
            total_duration += duration
            frame_accepted = False
            if result and result['labels_matched']:
                result.update({'time': duration})
                output.update({opfname: result})
                successful += 1
                frame_accepted = True
            else:
                print(' - No labels matched. Frame rejected')
            
            self.tc.interval_stop('Detect objects in frame',frame_accepted)
            
        print('Frames processed :', i)
        print('Overall Processing speed per image', (total_duration)/i)
        print('Total duration:',total_duration)
        cap.release()
        cv2.destroyAllWindows()
        return output,total_duration,successful
