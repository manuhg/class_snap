import os,sys
import json
from extractor import extract_frames

def get_input_data(dir_name,class_lalbels_file,allowed_file_types={'mp4':True,'avi':True}):
  with open(class_lalbels_file,"r") as labels_file:
    labels_str = labels_file.read()
    class_labels_to_filter = labels_str.replace('\n',',')
    class_labels_to_filter = list(filter(None,class_labels_to_filter.split(',')))
    input_video_files = list(filter(lambda f: allowed_file_types.get(f), os.listdir(dir_name)))
    return input_video_files,class_labels_to_filter
  print('error opening '+class_lalbels_file)

def compare_models():
  models = ['yolov2','yolov2-tiny','yolov3-tiny','ssdlite_mobilenet']
  results = {}
  for model in models:
    result = run_predictions(model) #TODO run_predictions()
    results.update(result)
    results_statistical_diff() #TODO results_statistical_diff()

def main():
  dest_dir = '.'
  if len(sys.argv) >= 3:
    #TODO ADD functionality to use zip files as i/p and o/p instead of dir later on
    dir_name = sys.argv[1]
    class_labels_file = sys.argv[2]
    if len(sys.argv)>3:
      dest_dir = sys.argv[3]

    input_video_files,class_labels_to_filter = get_input_data(dir_name,class_labels_file)
    for input_video_file in input_video_files:
      prediction_stats = extract_frames(input_video_file,class_labels,dest_dir,show_popup=True)
      with open(input_video_file+'-prediction_stats.txt',"w+") as sf:
        sf.write(json.dump(prediction_stats))
    #compare_models() TODO
  else:
    print('Format:\npython main.py <directory/zip file containing input videos> <text file containing class labels to filter> <destination (optional) > ')

if __name__ == "main":
  main()