FROM pytorch/pytorch:2.6.0-cuda11.8-cudnn9-devel

RUN apt update --fix-missing
RUN apt install build-essential -y
RUN apt install ffmpeg libsm6 -y
RUN apt install vim -y
RUN apt install imagemagick -y
RUN apt install curl -y
RUN apt clean

RUN pip install uv
RUN pip install huggingface_hub

WORKDIR /workspace

COPY pyproject.toml .
COPY uv.lock .

RUN uv sync