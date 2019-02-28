from fileutils import *
class detectron:
  def __init__(self,model_name='RetinaNet'):
    self.model_name = model_name
    self.env = 'env/'
    self.name = 'Detectron'
    self.cocoapi = self.env+'/cocoapi'
    self.env_dir = self.env+self.name
    self.pretrained_models_dir = self.env+'pretrained/'
    self.model = {
        'weights_url':'https://dl.fbaipublicfiles.com/detectron/36769563/12_2017_baselines/retinanet_X-101-32x8d-FPN_1x.yaml.08_42_05.06JTK6vJ/output/train/coco_2014_train:coco_2014_valminusminival/retinanet/model_final.pkl',
        'weights_file':'retinanet_X-101-32x8d-FPN_1x.pkl',
        'weights':self.pretrained_models_dir+'retinanet_X-101-32x8d-FPN_1x.pkl'}
    self.model['cfg'] = 'configs/12_2017_baselines/retinanet_X-101-32x8d-FPN_2x.yaml'
    
    
  def prepare_env(self):
    exec_cmd('mkdir -p '+self.env)
    exec_cmd('git clone https://github.com/cocodataset/cocoapi.git '+self.cocoapi)
    exec_cmd('cd '+self.cocoapi+'/PythonAPI && make install')

    exec_cmd('git clone https://github.com/facebookresearch/detectron '+self.env_dir)
    exec_cmd('ln -s '+self.cocoapi+' '+self.env_dir+'cocoapi')
    exec_cmd('pip install -r '+self.env_dir+'/requirements.txt')
    exec_cmd('cd '+self.env_dir+' && make')
    exec_cmd('python '+self.env_dir+'/detectron/tests/test_spatial_narrow_as_op.py')
    exec_cmd('mkdir -p '+self.pretrained_models_dir)
    
    download_weights = True
    if os.path.isfile(self.model['weights']) and (os.stat(self.model['weights']).st_size/1048576)>1:
      print('weights file exists. size: ',(os.stat(self.model['weights']).st_size/1048576),'Mb')
      download_weights = False
      
    if download_weights:
      download_file(self.model['weights_url'],self.model['weights'])