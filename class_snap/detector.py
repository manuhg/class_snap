from __future__ import print_function
from models.ssd import ssd
from models.yolo import yolo


class detector:
    ''' Generic class to plug in any model wrapped into a class
        The class must expose methods as below:
        
        -> __init__(self, prepared=False,env_parent='models/env/')
        
        -> prepare(self, prepared=None)
        
        -> load()  
        
        -> detect(self, image, opfile, class_labels_to_filter, visualize=False) #once per file/frame
            
            output_dict = {'bounding_boxes':bboxes_converted,'labels_detected':labels_detected}
            returns: {'labels_matched': labels_matched, 'output': output_dict}
    '''
    
    models = {'ssd': {'name': 'ssd', 'variant': ''},
              'yolo': {'name': 'yolo', 'variant': 'v2'}}
    datasets = ['voc', 'imagenet', 'coco']
    dataset = datasets[-1]

    

    def __init__(self,model_name='ssd'):
        self.models['ssd']['model'] = ssd
        self.models['yolo']['model'] = yolo
        #model classes should expose 3 functions: prepare,load,detect
        self.model_name = model_name
        self.model = self.models[self.model_name]

    def get_model(self):
        return self.model['model']()