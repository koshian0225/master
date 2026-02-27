FROM nvidia/cuda:11.0.3-cudnn8-devel-ubuntu20.04
# FROM pytorch/pytorch:1.7.0-cuda11.0-cudnn8-devel
SHELL ["/bin/bash", "-c"]
RUN sed -i.bak -e "s%http://[^ ]\+%http://ftp.jaist.ac.jp/pub/Linux/ubuntu/%g" /etc/apt/sources.list
# RUN rm /etc/apt/sources.list.d/cuda.list
# RUN rm /etc/apt/sources.list.d/nvidia-ml.list
# RUN apt-key del 7fa2af80
RUN apt-get update && apt-get install -y --no-install-recommends wget ca-certificates python3-pip
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.0-1_all.deb
RUN dpkg -i cuda-keyring_1.0-1_all.deb
RUN rm -f /etc/apt/sources.list.d/cuda.list /etc/apt/sources.list.d/nvidia-ml.list || true
ENV DEBIAN_FRONTEND=noninteractive
RUN pip3 install torch==1.7.0+cu110 torchvision==0.8.1+cu110 torchaudio==0.7.0 -f https://download.pytorch.org/whl/torch_stable.html
RUN apt-get update && \
    apt-get install -y \
        git \
        python3-numpy-dev \
        libpcl-dev \
        libopencv-dev && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/

ENV CMAKE_ARGS="-DCMAKE_POLICY_VERSION_MINIMUM=3.5"    

RUN if [[ ! -f /usr/lib/x86_64-linux-gnu/libboost_numpy3.so ]]; then \
        ln -s /usr/lib/x86_64-linux-gnu/libboost_numpy3*.so /usr/lib/x86_64-linux-gnu/libboost_numpy3.so ; \
    fi; \
    if [[ ! -f /usr/lib/x86_64-linux-gnu/libboost_python3.so ]]; then \
        ln -s /usr/lib/x86_64-linux-gnu/libboost_python3*.so /usr/lib/x86_64-linux-gnu/libboost_python3.so ; \
    fi
RUN if [[ -f /opt/conda/bin/activate ]]; then \
        source /opt/conda/bin/activate && \
        conda install -c conda-forge scikit-build ninja h5py==2.10.0 argcomplete optuna optuna-dashboard openpyxl pymysql; \
    else \
        pip3 install scikit-build ninja h5py==2.10.0 argcomplete optuna optuna-dashboard openpyxl pymysql; \
    fi; \
    pip install --verbose git+https://github.com/shikishima-TasakiLab/pointsmap-python && \
    pip install git+https://github.com/shikishima-TasakiLab/h5dataloader && \
    pip install git+https://github.com/shikishima-TasakiLab/h5datacreator && \
    pip install adabelief-pytorch
RUN pip install zipp==3.8.0 &&\
    pip uninstall -y nvidia-tensorboard || true && \
    pip uninstall -y nvidia-tensorboard-plugin-dlprof || true && \
    pip uninstall -y tensorboard || true && \
    pip install tensorboard torch-tb-profiler
RUN pip install matplotlib && \
    pip install albumentations && \
    pip install numpy==1.21 && \
    pip install rich && \
    pip install Pillow==9.5.0 && \
    pip install --ignore-installed PyYAML && \
    pip install open3d
RUN echo "export PS1='\[\033[01;32m\]pmod_bpre@docker\[\033[00m\]:\[\033[01;34m\]\w\[\033[00m\]\$ '" >> /root/.bashrc