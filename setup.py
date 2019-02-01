from setuptools import setup,find_packages

REQUIRED_PACKAGES = ['Pillow>=1.0', 'Matplotlib>=2.1', 'Cython>=0.28.1','opencv-python>=3.4.3','tensorflow>=1.9']

with open("README.md", "r") as readme:
    long_description = readme.read()
setup(
     name='class_snap',  
     version='0.1',
     author="Manu Hegde",
     author_email="manuhegdev@gmail.com",
     description="A tool to extract frames containing a class label, with bouding boxes applied",
     long_description=long_description,
     long_description_content_type="text/markdown",
     url="https://github.com/manuhg/class_snap",
     install_requires=REQUIRED_PACKAGES,
     include_package_data=True,
     packages=[p for p in find_packages() if p.startswith('object_detection')],
     classifiers=[
         "Programming Language :: Python :: 3",
         "License :: OSI Approved :: MIT License",
         "Operating System :: OS Independent",
     ],
 )
