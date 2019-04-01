#configure a cloud instance with nvidia gpu to run class_snap. 
# this is valid only for ubuntu 16.04 LTS
#!/bin/sh
sudo apt-get install gnupg-curl
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/cuda-repo-ubuntu1604_10.0.130-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu1604_10.0.130-1_amd64.deb
sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/7fa2af80.pub
sudo apt-get update
wget http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1604/x86_64/nvidia-machine-learning-repo-ubuntu1604_1.0.0-1_amd64.deb
sudo apt install ./nvidia-machine-learning-repo-ubuntu1604_1.0.0-1_amd64.deb
sudo apt-get update

# Install NVIDIA Driver
# Issue with driver install requires creating /usr/lib/nvidia
sudo mkdir /usr/lib/nvidia
sudo apt-get install --no-install-recommends nvidia-410
# Reboot. Check that GPUs are visible using the command: nvidia-smi

# Install development and runtime libraries (~4GB)
sudo apt-get install --no-install-recommends \
    cuda-10-0 \
    libcudnn7=7.4.1.5-1+cuda10.0  \
    libcudnn7-dev=7.4.1.5-1+cuda10.0

sudo apt install libnvinfer5=5.0.2-1+cuda10.0
echo "PLEASE REBOOT THE SYSTEM AND run nvidia-smi"
# Install TensorRT. Requires that libcudnn7 is installed above.
sudo apt-get update \
    && sudo apt-get install nvinfer-runtime-trt-repo-ubuntu1604-5.0.2-ga-cuda10.0 \
    && sudo apt-get update && sudo apt-get install -y --no-install-recommends libnvinfer-dev=5.0.2-1+cuda10.0


wget https://repo.anaconda.com/archive/Anaconda3-2018.12-Linux-x86_64.sh
chmod +x Anaconda3-2018.12-Linux-x86_64.sh
./Anaconda3-2018.12-Linux-x86_64.sh
source ~/.bashrc
echo "PATH=$PATH:/usr/local/cuda/bin" >> ~/.bashrc
conda activate

#conda create -n py2 python=2.7
sudo apt install libopencv-dev -y
conda create -n py2 python=2.7 opencv keras scikit-learn tensorflow-gpu pillow matplotlib numpy future pytorch-nightly -c pytorch
conda create -n py3 python=3.5 opencv keras scikit-learn tensorflow-gpu pillow matplotlib numpy  pytorch-nightly -c pytorch