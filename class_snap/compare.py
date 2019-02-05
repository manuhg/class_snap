
def compare_models():
  models = ['yolov2','yolov2-tiny','yolov3-tiny','ssdlite_mobilenet']
  results = {}
  for model in models:
    result = run_predictions(model) #TODO run_predictions()
    results.update(result)
    results_statistical_diff() #TODO results_statistical_diff()
