import time
import cv2
from detector import detector

def extract_frames(input_file,class_labels,dest_dir='.',interval=None): #interval if specified should be in terms of seconds
  interval = interval * 1000 # convert to milliseconds
  cap = cv2.VideoCapture(input_file)
  labels_matched = {} #format: File name : [list of matched labels]
  if (cap.isOpened()== False): 
    print("Error opening video file",input_file)
  i,count = 0,0
  fps = video.get(cv2.CAP_PROP_FPS)
  input_file_name =  '.'.join(input_file.split('.')[:-1])
  
  t1 = time.time()
  if dest_dir[-1] != '/':
      dest_dir = dest_dir + '/'

  while(cap.isOpened()):
    if interval:
      cap.set(cv2.CAP_PROP_POS_MSEC,interval)
    ret, frame = cap.read()
    i+=1
    if ret == True:
      opfname = input_file_name+'-'+str(int(i/fps))+':'+str(i%fps)+'.jpg'
      class_labels_matched = detector.get_detections(frame,class_labels,dest_dir+opfname)
      count+=1
      if class_labels_matched:
        labels_matched.update({opfname:class_labels_matched})
    else:
      break
  t2 = time.time()
  print('Overall Processing speed:',(t2-t1)/i)
  print('Frames with detections:{0}/{1}'%(count,i))
  cap.release()
  cv2.destroyAllWindows()
  return labels_matched