from __future__ import print_function
import json
from fileutils import unannotate_all
from sklearn.metrics import confusion_matrix
from extractor import extractor


# def compare_models():
#   models = ['yolov2','yolov2-tiny','yolov3-tiny','ssdlite_mobilenet']
#   results = {}
#   for model in models:
#     result = run_predictions(model) #TODO run_predictions()
#     results.update(result)
#     results_statistical_diff() #TODO results_statistical_diff()
def compare_models(input_video_files,class_labels_to_filter,interval,zip_name,visualize=False,models = {'yolo':['yolov2','yolov2-tiny','yolov3-tiny'],'detectron':['detectron']}):
  loaded_models = {}
  model_times = {}
  for model_name in list(models.keys()):
    for model_variant in models[model_name]:
      loaded_models[model_name+model_variant] = extractor(model_name=model_name,model_variant=model_variant,load=True)
  
  for model_name in list(models.keys()):
    for model_variant in models[model_name]:
      for input_video_file in input_video_files:
        print(input_video_file,class_labels_to_filter,interval,zip_name)
        output,total_duration = loaded_models[model_name+model_variant].process(input_video_file,class_labels_to_filter,interval,zip_name=zip_name,visualize=visualize)
        model_times[model_variant] = total_duration
  return model_times

class labelencoder:
  def __init__(self,classes_lst):
    self.classes_lst = classes_lst
    self.encode()
  
  def encode(self):
    encoded_classes = {}
    classes_lst = self.classes_lst
    for i in range(len(classes_lst)):
      c = classes_lst[i]
      encoded_classes[c] = i
    self.encoded_classes = encoded_classes
  
  def transform(self,vals_lst):
    return [ self.encoded_classes.get(v) for v in vals_lst ]

def compare(ground_truth_ann,annotated_output):
  gt = convallforcomparison(unannotate_all(ground_truth_ann))
  po = convallforcomparison(unannotate_all(annotated_output))
  res = {}
  if len(gt.keys()) != len(po.keys()):
    print("Error: mismatch of length in ground truth and annotation")
  
  for k in gt.keys():
    evals = calculate_metrics(po[k],gt[k])
    evals.update(calculate_accuracy(po[k],gt[k]))
    res.update({k:evals})
  return res

def convforcomparison(unann_item):
  fname, output = unann_item
  return {fname:output['output']['labels_detected']}

def convallforcomparison(annotated_output):
  comp_dict = {}
  for d in list(map(convforcomparison,annotated_output.items())):#unannotate_all(annotated_output).items())):
    comp_dict.update(d)
  return comp_dict

def get_ground_truth_ann(filename): # a file containing list of annotation dicts per file
  ground_truth_ann = None,None
  try:
    with open(filename,'r') as gtf:
      ground_truth_ann = json.load(gtf)
  except Exception as e:
    print('Error getting ground truth\n',e)
  return ground_truth_ann

def calculate_accuracy(predicted,ground_truth):
  return  {'accuracy':float(len(set(ground_truth) & set(ground_truth)))/float(len(set(ground_truth)))}


def calculate_metrics(predicted,ground_truth):#once per file
  len_diff = len(predicted)-len(ground_truth)
  len_diff1 = -1 * len_diff if len_diff<0 else len_diff
  pad = [ '__NAP__' for i in range(len_diff1)]  
  if len_diff>0: #many false positives i.e more predictions that actual
      ground_truth = ground_truth + pad
  elif len_diff<0:
      predicted = predicted + pad

  #THIS FUNCTION USES __NAP__ - NOT A PREDICTION TO FILL GAPS WHEREVER NECESSARY
  classes = list(set(ground_truth+predicted))
  le = labelencoder(classes)
  
  ground_truth = le.transform(ground_truth)
  predicted = le.transform(predicted)
  cm = confusion_matrix(ground_truth,predicted)
  colsums = [sum(cm[:,i]) for i in range(cm.shape[0]) ]
  rowsums = [sum(cm[i,:]) for i in range(cm.shape[1]) ]
  metrics ={}
  for i in range(len(classes)):
    precision = float(cm[i][i])/float(colsums[i]) if colsums[i]!=0 else -1
    recall = float(cm[i][i])/float(rowsums[i]) if rowsums[i]!=0 else -1
    f1 = 2 * float(precision * recall) / float(precision + recall) if (precision + recall) !=0 else -1
    metrics[classes[i]] = {'precision':precision,'recall':recall,'f1':f1}
  return metrics

def compare_outputs(ground_truth_ann,model_output_ann):
  ground_truth_labels = unannotate_all(ground_truth_ann) #for all files
  model_output_labels = unannotate_all(model_output_ann) #for all files
  if ground_truth_labels.keys() != model_output_labels.keys():
    print('Data mismatch!')
    return

# a,b=unannotate_all(extractor_.annotated_output),extractor_.output
# convallforcomparison(b)
#10.16.18.47, IMDB WIKI face dataset