# file_utils.py
import os
import six.moves.urllib as urllib
import tarfile


def exec_cmd(cmdstr, echo=True):
    print(os.popen(cmdstr).read() if True else '')


def download_file(url, filename=None):
    fn = (str(' -O '+filename) if filename else ' ')
    exec_cmd('wget -nc '+url+fn)
    return filename if filename else url.split('/')[-1]

# def download_file(url,filename):
#   opener = urllib.request.URLopener()
#   opener.retrieve(url, filename)


def create_zip(zip_name, file_names):
    exec_cmd('zip -r '+zip_name+' '+' '.join(file_names))


def extract_file_from_tar(tar_file, filename_to_ext):
    tar_file = tarfile.open(tar_file)
    for file in tar_file.getmembers():
        file_name = os.path.basename(file.name)
        if filename_to_ext in file_name:
            tar_file.extract(file, os.getcwd())


def load_image_into_numpy_array(self, image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)
