FROM anibali/pytorch:1.5.0-cuda10.2

# WORKDIR
USER root

COPY train.py .
COPY Pipfile* ./

ADD train train
ADD val val

RUN apt-get update && \
    apt-get install --no-install-recommends -y \
    gcc swig python3.7 python3-setuptools python3-pip python3-dev libgl1-mesa-glx libglib2.0-0 && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

RUN pip3 install --upgrade pip
RUN pip3 install sagemaker-training pipenv

# Install Pipfile requirements
ENV LC_ALL C.UTF-8
ENV LANG C.UTF-8
RUN pipenv lock --requirements > requirements.txt
RUN pip install -r requirements.txt

# Remove for Sagemaker
ENV SM_MODEL_DIR .
ENV SM_CHANNEL_TRAINING .

RUN ls
CMD python train.py --data-dir . --model-dir . --action final_layer --epochs 20