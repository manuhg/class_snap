from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
ext_modules = [
    Extension("mymodule1",  ["class_snap/fuse_annotations.py"]),
    Extension("mymodule2",  ["class_snap/extractor.py"]),
    Extension("mymodule2",  ["class_snap/compare.py"]),
    Extension("mymodule2",  ["class_snap/detector.py"]),
    Extension("mymodule2",  ["class_snap/fileutils.py"]),
    Extension("mymodule2",  ["class_snap/models/ssd.py"]),
    Extension("mymodule2",  ["class_snap/models/yolo.py"]),
    Extension("mymodule2",  ["class_snap/models/detectron_fb.py"]),
]
setup(
    name = 'class_snap',
    cmdclass = {'build_ext': build_ext},
    ext_modules = ext_modules
)