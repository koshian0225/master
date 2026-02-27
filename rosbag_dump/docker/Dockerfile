FROM ros:noetic-ros-core-focal
RUN perl -p -i.bak -e 's%(deb(?:-src|)\s+)https?://(?!archive\.canonical\.com|security\.ubuntu\.com)[^\s]+%$1http://linux.yz.yamagata-u.ac.jp/pub/linux/ubuntu/archives/%' /etc/apt/sources.list

ENV NVIDIA_VISIBLE_DEVICES ${NVIDIA_VISIBLE_DEVICES:-all}
ENV NVIDIA_DRIVER_CAPABILITIES=all
ENV DEBIAN_FRONTEND noninteractive

RUN apt-key del 3bf863cc && \
    apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3-pip \
        sudo \
        curl \
        wget \
        unar \
        bash-completion \
        libfreetype6-dev \
        libxft-dev \
        libjpeg62 \
        libjpeg-dev \
        libgl1-mesa-dev \
        libglib2.0-0 \
        software-properties-common \
        tzdata \
        git \
        nano \
        ffmpeg \
        libpcl-dev \
        libopencv-dev \
        locales \
        ros-noetic-cv-bridge \
        build-essential \
        python3-dev && \
    apt-get --fix-broken install && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN pip3 install --verbose \
        numpy \
        open3d \
        opencv-python \
        matplotlib \
        scikit-image \
        imageio \
        rich \
        Pillow \
        numpy-quaternion \
        cv_bridge \
        pydantic \
        pypcd4 \
        rosbags

WORKDIR /workspace

ARG UID
ARG USER_NAME
ARG HOSTNAME
ENV HOME=/home/${USER_NAME}

RUN localedef -f UTF-8 -i ja_JP ja_JP.utf8 && \
    locale-gen ja_JP.UTF-8 && \
    id -u ${USER_NAME} >/dev/null 2>&1 || adduser --uid ${UID} --disabled-password --gecos '' --shell /bin/bash ${USER_NAME} && \
    usermod -aG sudo ${USER_NAME} && \
    chown -R ${USER_NAME}:${USER_NAME} /workspace && \
    mkdir -p /usr/bin/shared && \
    mkdir -p /etc/sudoers.d/ && \
    echo "${USER_NAME} ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/90-${USER_NAME} && \
    chmod 777 /home/${USER_NAME} && \
    echo "export PS1='\[\033[01;32m\]${USER_NAME}@${HOSTNAME}\[\033[00m\]:\[\033[01;34m\]\w\[\033[00m\]\$ '" >> /etc/bash.bashrc && \
    echo "PYTHONIOENCODING=utf-8" >> /etc/bash.bashrc && \
    echo 'alias pip="pip3"' >> /etc/bash.bashrc && \
    mkdir -p /workspace/.bash_histories && \
    touch /workspace/.bash_histories/.bash_history_${USER_NAME} && \
    chmod 600 /workspace/.bash_histories/.bash_history_${USER_NAME}

ENV LANG ja_JP.UTF-8
ENV LANGUAGE ja_JP:ja
ENV LC_ALL=ja_JP.UTF-8
ENV HISTFILE=/workspace/.bash_histories/.bash_history_${USER_NAME}
ENV PATH $PATH:/home/${USER_NAME}/.local/bin
ENV PATH $PATH:/usr/bin/shared
ENV DEBIAN_FRONTEND newt
