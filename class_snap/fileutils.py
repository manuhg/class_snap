##fileutils.py
import os
import json
import six.moves.urllib as urllib
import tarfile
import zipfile

def exec_cmd(cmdstr,echo=True):
  print(os.popen(cmdstr).read() if echo else '',end='')
  return True
  
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

# def download_file(url,filename):
#   opener = urllib.request.URLopener()  
#   opener.retrieve(url, filename)
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
  fname = parent_dir+dct['annotation']['data_filename']
  fname = '.'.join(fname.split('.')[:-1])+'.json'
  try:
    with open(fname,'w+') as f:
      json.dump(dct,f)
    return fname
  except Exception as e:
    print('Error writing to '+fname)
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
  return success,failure
  #result = [{dct['annotation']['data_filename']:save_as_json(dct)} for dct in annotated_output ]
  #return result,list(filter(lambda x: type(tuple(x.items())[0][-1]) is tuple ,gg))