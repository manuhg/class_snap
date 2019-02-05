import os,sys
class Predictor:
    datasets = {'voc': 'voc', 'imagenet': 'imagenet', 'coco': {
        'name': 'coco', 'url': 'https://github.com/cocodataset/cocoapi.git','prepare':self.prepare_coco}}
    dataset = datasets['coco']
    def execcmd(self,command_str,echo=True):
        opstr = os.popen(command_str).read()
        if echo:
            print(opstr)
        #return opstr

    def prepare_coco(self):
        dataset=self.datasets['coco']
        self.execcmd('git clone '+dataset['url']+' '+dataset['name'])



    

    def prepare_dataset(self, dataset=self.dataset):
        git clone https: // github.com/cocodataset/cocoapi.git
    cd cocoapi/PythonAPI
