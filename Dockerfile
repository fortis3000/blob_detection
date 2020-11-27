# Use it on debugging or service stages
# Examples:
# docker build -r tf_segmentation .
# nvidia-docker run --gpus all -it --rm --name segmentation_models \
#	-v /home/sad/bulk_detection:/bulk_detection \
#	tf_segmentation

FROM tensorflow/tensorflow:2.1.2-gpu

RUN apt-get install -y libgl1-mesa-glx
RUN pip install segmentation-models opencv-python
RUN pip install --upgrade tensorflow-gpu
RUN pip install albumentations
RUN pip install hyperopt