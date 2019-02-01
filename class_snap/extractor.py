import time
import cv2
from predictor import predictor

def extract_frames(input_file,class_labels,dest_dir='.',show_popup=False):
  cap = cv2.VideoCapture(input_file)
  labels_matched = {} #format: File name : [list of matched labels]
  if (cap.isOpened()== False): 
    print("Error opening video file",input_file)
  i,count=0,0
  fps = video.get(cv2.CAP_PROP_FPS)
  input_file_name =  '.'.join(input_file.split('.')[:-1])
  t1 = time.time()
  if dest_dir[-1] != '/':
      dest_dir = dest_dir + '/'
  while(cap.isOpened()):
    ret, frame = cap.read()
    i+=1
    if ret == True:
      frame,class_labels_matched = predictor.get_predictions(frame,class_labels)
      if class_labels_matched:
        opfname = input_file_name+'-'+str(int(i/fps))+':'+str(i%fps)+'.jpg'
        labels_matched.update({opfname:class_labels_matched})
        cv2.imwrite(dest_dir+opfname,frame)
        if show_popup:
            cv2.imshow('Frame',frame)      
    else:
      break
  t2 = time.time()
  print('Processing speed:',(t2-t1)/i)
  cap.release()
  cv2.destroyAllWindows()
  return labels_matched