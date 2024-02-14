FROM nvcr.io/nvidia/l4t-pytorch:r34.1.1-pth1.11-py3

ARG USER_ID
ARG GROUP_ID
RUN addgroup --gid $GROUP_ID skipformer
RUN adduser --disabled-password --gecos '' --uid $USER_ID --gid $GROUP_ID skipformer
RUN usermod -aG video skipformer

ARG DEBIAN_FRONTEND=noninteractive

RUN apt update
RUN apt install -y git libsndfile1-dev tesseract-ocr espeak-ng python3 python3-pip ffmpeg
RUN python3 -m pip install --no-cache-dir --upgrade pip

RUN python3 -m pip install --no-cache-dir transformers==4.25.1 datasets==2.8.0 tokenizers==0.13.2 accelerate==0.15.0 evaluate==0.4.0

RUN python3 -m pip uninstall -y tensorflow flax

RUN python3 -m pip install --no-cache-dir torch-scatter -f https://data.pyg.org/whl/torch-$(python3 -c "from torch import version; print(version.__version__.split('+')[0])")+cu113.html
RUN python3 -m pip install --no-cache-dir git+https://github.com/facebookresearch/detectron2.git pytesseract
RUN python3 -m pip install -U "itsdangerous<2.1.0"

RUN python3 -m pip install --no-cache-dir wandb
RUN python3 -m pip install --no-cache-dir scipy pyarrow==12.0.0 fsspec==2023.9.2

RUN apt-get autoremove -y
RUN apt-get clean

USER skipformer

