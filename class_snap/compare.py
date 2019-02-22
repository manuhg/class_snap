import json
from fileutils import unannotate_all
# def compare_models():
#   models = ['yolov2','yolov2-tiny','yolov3-tiny','ssdlite_mobilenet']
#   results = {}
#   for model in models:
#     result = run_predictions(model) #TODO run_predictions()
#     results.update(result)
#     results_statistical_diff() #TODO results_statistical_diff()
# print(extractor_.output['input_video-0090s.jpg']['output'])
# print(unannotate_all(extractor_.annotated_output)['input_video-0090s.jpg']['output'])
def convforcomparison(unann_item):
  fname, output = unann_item
  return {fname:output['output']['labels_detected']}

def convallforcomparison(annotated_output):
  comp_dict = {}
  for d in list(map(convforcomparison,annotated_output.items())):#unannotate_all(annotated_output).items())):
    comp_dict.update(d)
  return comp_dict

def get_ground_truth(filename): # a file containing list of annotation dicts per file
  ground_truth_ann,ground_truth = None,None
  try:
    with open(filename,'r') as gtf:
      ground_truth_ann = json.load(gtf)
      ground_truth = unannotate_all(ground_truth_ann)
  except Exception as e:
    print('Error getting ground truth\n',e)
  return ground_truth_ann,ground_truth

def compare_outputs(ground_truth_ann,model_output_ann):
  ground_truth_labels = unannotate_all(ground_truth_ann)



# a,b=unannotate_all(extractor_.annotated_output),extractor_.output
# convallforcomparison(b)