# detector.py
from models.ssd import ssd
#from models.yolo import yolo


class detector:
    models = {'ssd': {'name': 'ssd', 'variant': ''},
              'yolo': {'name': 'yolo', 'variant': 'v2'}}
    datasets = ['voc', 'imagenet', 'coco']
    dataset = datasets[-1]

    def __init__(self,model_name):
        self.models['ssd']['model'] = ssd
        #self.models['yolo']['model'] = yolo
        self.model_name = model_name if model_name else 'ssd'
        self.model = self.models[self.model_name]

    def get_model(self):
        return self.model['model']()