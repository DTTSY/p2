# hash:sha256:b7949b1a25dd95630f45c7ff5057cb812f60e6504b90d9021db43408fe0d3869
FROM registry.codeocean.com/codeocean/miniconda3:4.3.30-python3.6.3-ubuntu16.04

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        gcc=4:5.3.1-1ubuntu1 \
        libevent-dev=2.0.21-stable-2ubuntu0.16.04.1 \
        libffi-dev=3.2.1-4 \
        openssl=1.0.2g-1ubuntu4.15 \
        python3-dev=3.5.1-3 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install -U --no-cache-dir --upgrade-strategy=only-if-needed \
    gevent==1.4.0 \
    numpy==1.16.4 \
    pandas==0.25.3 \
    psutil==5.4.0 \
    sklearn==0.0 \
    tables==3.4.2

COPY postInstall /
RUN /postInstall
