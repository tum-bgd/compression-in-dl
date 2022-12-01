#FROM tensorflow/tensorflow@sha256:1d0736e46ae9a961c2111394a43e0bfd266e6151a90d613b6f86229cf01e40e5
#FROM tensorflow/tensorflow:a34c2420739cd5a7b5662449bc21eb32d3d1c98063726ae2bd7db819cc93d72f
FROM tensorflow/tensorflow:2.4.3-gpu-jupyter


ARG DEBIAN_FRONTEND=noninteractive
RUN rm /etc/apt/sources.list.d/cuda.list
RUN rm /etc/apt/sources.list.d/nvidia-ml.list
RUN apt-get update
RUN apt-get -y install libhdf5-dev python3-h5py python3-opencv
RUN pip3 install Pillow;
RUN pip3 install scipy;
RUN pip3 install numpy
RUN pip3 install h5py
RUN pip3 install opencv-python
RUN pip3 install rasterio sklearn
RUN pip3 install tensorflow_io

RUN ls -la /sbin/ | grep ldconfig
RUN /sbin/ldconfig

RUN mkdir -p /source
WORKDIR /tf