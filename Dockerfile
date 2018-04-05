FROM gcr.io/tensorflow/tensorflow:1.2.0-gpu
WORKDIR .
ADD tensorflow tensorflow
ADD data data