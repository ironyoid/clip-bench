ARG BASE_IMAGE=nvcr.io/nvidia/pytorch:23.12-py3
FROM ${BASE_IMAGE} AS base

ENV DEBIAN_FRONTEND=noninteractive

# Setup non-root user
ARG USERNAME=user
ARG UID=1000
ARG GID=$UID
ARG PW=$USERNAME

RUN groupadd --gid $GID $USERNAME \
    && useradd --uid $UID --gid $GID -m $USERNAME \
    && apt-get update \
    && apt-get install -y sudo \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME

# Install dependencies
RUN apt update && apt install -q -y --no-install-recommends nano && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*
COPY requirements.txt /home/${USERNAME}/requirements.txt
RUN pip install -r /home/${USERNAME}/requirements.txt
WORKDIR /home/${USERNAME}
USER $USERNAME