from __future__ import print_function
import os
import json
import six.moves.urllib as urllib
import tarfile
import zipfile
import numpy as np

def import_pytube():
    try:
        global YouTube
        from pytube import YouTube
        return True
    except Exception as e:
        print('Unable to import pytube')

def download_youtube_video(url):
    if not import_pytube():
        print('instaling pytube')
        exec_cmd('pip install pytube')
        if not import_pytube():
            print('Fatal! unable to install or import pytube!. Quitting.')
            return False
    
    print('Downloading the video at url ',url)
    try:
      yt = YouTube(url)#.streams.first().download()
      filepath = yt.streams.first().download()
      filename = filepath.split('/')[-1]
      return filename
    except Exception as e:
      print('Video not found at url\n',e)

def exec_cmd(cmdstr,echo=True):
  print(os.popen(cmdstr).read() if echo else '',end='')

def download_file(url,filename=None,nc=False):#nc no clobber i.e dont download if file exists
  if nc:
    fname = filename if filename is not None else  url.split('/')[-1]
    if os.path.isfile(fname):
      print('File ',fname,'exists. Skipping download')
      return fname
  print('Downloading from',url)
  fn = (str(' -O '+filename) if filename else ' ')
  exec_cmd('wget -nc '+url+fn)
  return filename if filename else url.split('/')[-1]

def download_file_urllib(url,filename):
  print('Downloading from',url,' to ',filename)
  opener = urllib.request.URLopener()
  opener.retrieve(url, filename)
  print('Downloaded file size: ',os.stat(filename).st_size/1048576,'Mb')

def create_zip(zip_name,file_names):
  print('Creating zip file ',zip_name)
  exec_cmd('zip -r '+zip_name+' '+' '.join(file_names))

def extract_file_from_tar(tar_file,filename_to_ext,dest_dir):
  print('Trying to extract',filename_to_ext,'from',tar_file)
  tar_file = tarfile.open(tar_file)
  for file in tar_file.getmembers():
    file_name = os.path.basename(file.name)
    if filename_to_ext in file_name:
      tar_file.extract(file, dest_dir)
      
def load_image_into_numpy_array(self,image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)
 
def unannotate_image(ann_dict): #once per file
  #annotation format to gof - general output format the way yolo/ssd give their o/p
  #output_dict = {'output':{fname:{}}, 'labels_matched':None, 'time':-1} #dict in gof
  fname = ann_dict['annotation']['data_filename']
  bbox_anns = ann_dict['annotation']['data_annotation']['bounding_box']
  bbox_annotations = {}
  bbk,lbk = 'bounding_boxes','labels_detected'
  bbox_annotations[bbk] = []
  bbox_annotations[lbk] = []
  bbk_lst,lbk_lst  = bbox_annotations[bbk], bbox_annotations[lbk]
  for ba in bbox_anns:
    lbk_lst.append(ba['classification_label'])
    coords =  list(map(lambda x:  tuple(map(lambda v: int(v), x.split(','))),ba['point_2D'] )) # "5,6","3,4" = > ((5,6),(3,4))
    bbk_lst.append(list( coords[0]+coords[1] ))
  return  {fname:{'labels_matched': [], 'output': {bbk:bbk_lst,lbk:lbk_lst} , 'time':-1}}

def unannotate_all(annotated_output):
  opd = {}
  for d in list(map(unannotate_image,annotated_output)):
    opd.update(d)
  return opd
  
import json
def bbox_annotations(output): # call once per file
  bbox_annotations=[]
  bbk,lbk = 'bounding_boxes','labels_detected'
  tmp = output['output']
  for i in range(len(tmp[bbk])):
    coordinates = [str(val) for val in tmp[bbk][i]]
    pt1 = ','.join(coordinates[:2])
    pt2 = ','.join(coordinates[2:])
    bbox_annotations.append({'classification_label':tmp[lbk][i],'point_2D':[pt1,pt2]})
  return bbox_annotations

def annotate_image(output_item):  # call once per file
  fname, output_dict = output_item
  annotations = {'annotation': {'data_filename': fname, 'data_type': 'image', 'data_annotation': {'bounding_polygon': [], 'bounding_box': ''}}}
  annotations['annotation']['data_annotation']['bounding_box'] = bbox_annotations(output_dict)
  return annotations

def save_as_json(dct,parent_dir='./'): # call once per file
  json_fname = dct['annotation']['data_filename'].split('/')[-1]
  fname = parent_dir+json_fname
  fname = '.'.join(fname.split('.')[:-1])+'.json'
  fname = fname.replace("\\ ", " ")
  try:
    with open(fname,'w+') as f:
      json.dump(dct,f)
    return fname
  except Exception as e:
    print('Error writing to '+fname,'\n',e)
  return ('Failed!',fname)
  
def save_as_annotations(output,opdir='./'):
  print('Saving annotations to file')
  opdir = opdir if opdir[-1]=='/' else opdir+'/' #add trailing / if not present
  annotated_output = list(map(annotate_image,list(output.items())))
  success,failure=[],[]
  for dct in annotated_output:
    fname = save_as_json(dct,opdir)
    lst = failure if type(fname) is tuple else success
    lst.append(fname)
  return success,failure,annotated_output
  #result = [{dct['annotation']['data_filename']:save_as_json(dct)} for dct in annotated_output ]
  #return result,list(filter(lambda x: type(tuple(x.items())[0][-1]) is tuple ,gg))

def save_dicts_to_file(data_dict,filename):
        file_dict = None
        try:
            f = open(filename,'r+')
            file_dict = json.loads(f.read())
        except Exception as e:
                print('Cannot open file ',filename,'\nCreating new file')
                f = open(filename,'w+')
        finally:
            f.close()
      
        index = 0
        df = {}
        df[index] = data_dict
        if file_dict:
          index = len(file_dict) #+1 is not needed since index begins from zero
          df = file_dict
        try:
            df[index] = data_dict
        except Exception as e:
            print('Error while merging dicts',e)
            df[index] = data_dict
        with open(filename,'w+') as f:
            json.dump(df,f)
