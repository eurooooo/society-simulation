FROM cuda:12.1.6-devel-ubuntu22.04

LABEL maintainer="zaemyung@gmail.com" \
      org.opencontainers.image.source="https://github.com/zaemyung/dockerfiles"

ENV LANG=C.UTF-8
ENV SHELL=/bin/bash

ENV LD_LIBRARY_PATH /usr/local/cuda/extras/CUPTI/lib64:$LD_LIBRARY_PATH

RUN apt-get update && apt-get install -y \
        libblas-dev \
        liblapack-dev \
        graphviz \
    && rm -rf /var/lib/apt/lists/*
FROM zaemyung/cuda:12.1-python3.11

LABEL maintainer="zaemyung@gmail.com" \
      org.opencontainers.image.source="https://github.com/zaemyung/dockerfiles"

ENV LANG=C.UTF-8
ENV SHELL=/bin/bash

ENV LD_LIBRARY_PATH /usr/local/cuda/extras/CUPTI/lib64:$LD_LIBRARY_PATH

RUN apt-get update && apt-get install -y \
        libblas-dev \
        liblapack-dev \
        graphviz \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir --upgrade \
        pip \
        setuptools \
        wheel

