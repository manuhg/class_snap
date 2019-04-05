from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
ext_modules = [
#    Extension("fuse_annotations",  ["class_snap/fuse_annotations.py","class_snap/fileutils.py"]),
    #Extension("extractor",  ["class_snap/extractor.py","class_snap/detector.py","class_snap/fileutils.py","class_snap/models/ssd.py","class_snap/models/yolo.py"]),
    Extension("class_snap",  ["class_snap/main.py","class_snap/compare.py","class_snap/extractor.py","class_snap/detector.py","class_snap/fileutils.py","class_snap/models/ssd.py","class_snap/models/yolo.py","class_snap/models/detectron_fb.py"])
#    Extension("compare",  ["class_snap/compare.py","class_snap/fileutils.py"]),
#    Extension("detector",  ["class_snap/detector.py"]),
#    Extension("fileutils",  ["class_snap/fileutils.py"]),
#    Extension("ssd",  ["class_snap/models/ssd.py"]),
#    Extension("yolo",  ["class_snap/models/yolo.py"]),
#    Extension("detectron_fb",  ["class_snap/models/detectron_fb.py"])
]
setup(
    name = 'class_snap',
    cmdclass = {'build_ext': build_ext},
    ext_modules = ext_modules
)
