# Use it for production
FROM tensorflow/tensorflow:2.1.2-gpu

RUN apt-get install -y libgl1-mesa-glx

COPY . /bulk_detection

WORKDIR /bulk_detection

RUN pip install -r requirements_docker.txt && rm -f requirements_docker.txt
