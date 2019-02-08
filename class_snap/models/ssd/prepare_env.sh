git clone --recursive https://github.com/tensorflow/models.git md
git clone https://github.com/cocodataset/cocoapi.git
cd cocoapi/PythonAPI && make && cp -rv pycocotools ../../md/research/
cd md/research && protoc object_detection/protos/*.proto --python_out=.
cd md/research && python setup.py install
cd md/research/slim && python setup.py install
cd md/research && python object_detection/builders/model_builder_test.py
mv -v md/research/* ./
#mv md/research/object_detection ./
mv -v md/research/setup.py ./
rm -rf md
ls
python object_detection/builders/model_builder_test.py