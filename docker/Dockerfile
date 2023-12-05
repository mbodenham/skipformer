FROM nvcr.io/nvidia/pytorch:22.02-py3

ARG USER_ID
ARG GROUP_ID
RUN addgroup --gid $GROUP_ID user
RUN adduser --disabled-password --gecos '' --uid $USER_ID --gid $GROUP_ID user

#--------------START--------------#

ARG DEBIAN_FRONTEND=noninteractive

RUN apt update
RUN apt install -y git libsndfile1-dev tesseract-ocr espeak-ng python3 python3-pip ffmpeg
RUN python3 -m pip install --no-cache-dir --upgrade pip

RUN python3 -m pip install --no-cache-dir transformers==4.25.1 datasets==2.8.0 tokenizers==0.13.2 accelerate==0.15.0 evaluate==0.4.0

RUN python3 -m pip uninstall -y tensorflow flax

RUN python3 -m pip install --no-cache-dir torch-scatter -f https://data.pyg.org/whl/torch-$(python3 -c "from torch import version; print(version.__version__.split('+')[0])")+cu113.html
RUN python3 -m pip install --no-cache-dir git+https://github.com/facebookresearch/detectron2.git pytesseract
RUN python3 -m pip install -U "itsdangerous<2.1.0"

RUN python -m pip install --no-cache-dir wandb
RUN python -m pip install pyarrow==12.0.0

RUN apt install -y iputils-ping net-tools

RUN apt-get autoremove -y
RUN apt-get clean


#RUN echo "10.150.21.75 ids-lab-75" >> /etc/hosts
#RUN echo "10.150.21.79 ids-lab-79" >> /etc/hosts
#---------------END---------------#

USER user

RUN wandb login f9c5e37b4b3a89889dff64c776a070692a1d6201
