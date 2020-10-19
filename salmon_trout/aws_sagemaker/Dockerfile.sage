FROM anibali/pytorch:1.5.0-cuda10.2

# WORKDIR
WORKDIR /opt/ml/code/

COPY . .

USER root

RUN sudo apt-get update && \
    sudo apt-get install --no-install-recommends -y \
    gcc swig python3.7 python3-setuptools python3-pip python3-dev libgl1-mesa-glx libglib2.0-0 && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

RUN pip3 install --upgrade pip
RUN pip3 install sagemaker-training pipenv

# Install Pipfile requirements
ENV LC_ALL C.UTF-8
ENV LANG C.UTF-8
RUN pipenv lock --requirements > requirements.txt
RUN pip install -r requirements.txt

ENV SAGEMAKER_PROGRAM train.py
