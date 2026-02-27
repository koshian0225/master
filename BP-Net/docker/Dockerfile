FROM nvidia/cuda:12.1.1-devel-ubuntu22.04
SHELL ["/bin/bash", "-c"]
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    wget curl git ca-certificates \
    gcc g++ build-essential \
    python3-dev python3-pip python3-numpy \
    libboost-python-dev libboost-numpy-dev \
    libhdf5-dev libopencv-dev \
    libvtk9-dev libeigen3-dev libglew-dev \
    libtbb-dev libopenni-dev libopenni2-dev \
    freeglut3-dev libgl1-mesa-dev libglu1-mesa-dev && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

ENV CONDA_DIR=/opt/conda
ENV PATH=$CONDA_DIR/bin:$PATH
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    bash Miniconda3-latest-Linux-x86_64.sh -b -p $CONDA_DIR && \
    rm Miniconda3-latest-Linux-x86_64.sh

COPY environment.yml .
RUN conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main && \
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r && \
    conda update -n base -c defaults conda -y && \
    conda env create -f environment.yml && \
    conda clean -a -y

RUN sed -i 's/return caster\.operator typename make_caster<T>::template cast_op_type<T>();/return caster;/' \
$CONDA_DIR/envs/bp/lib/python3.9/site-packages/torch/include/pybind11/cast.h

RUN $CONDA_DIR/envs/bp/bin/pip install --upgrade pip && \
    $CONDA_DIR/envs/bp/bin/pip install scikit-build ninja cython==0.29.36 && \
    wget https://github.com/Kitware/CMake/releases/download/v3.31.2/cmake-3.31.2-linux-x86_64.sh && \
    bash cmake-3.31.2-linux-x86_64.sh --prefix=$CONDA_DIR/envs/bp --skip-license && \
    rm cmake-3.31.2-linux-x86_64.sh

RUN conda install -n bp -c conda-forge boost=1.72 pcl boost-cpp h5py

RUN git clone https://github.com/open-source-parsers/jsoncpp.git && \
    cd jsoncpp && mkdir build && cd build && \
    $CONDA_DIR/envs/bp/bin/cmake .. -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON -DCMAKE_INSTALL_PREFIX=/usr/local && \
    make -j$(nproc) && make install && \
    cd ../.. && rm -rf jsoncpp

RUN git clone http://tfsv.tasakilab:5051/git/fukuta/pointsmap-python.git && \
    source $CONDA_DIR/etc/profile.d/conda.sh && \
    conda activate bp && \
    pip install ./pointsmap-python && \
    rm -rf pointsmap-python

RUN git clone http://tfsv.tasakilab:5051/git/fukuta/h5dataloader.git && \
    source $CONDA_DIR/etc/profile.d/conda.sh && \
    conda activate bp && \
    HDF5_DIR=$CONDA_DIR/envs/bp pip install ./h5dataloader && \
    rm -rf h5dataloader

RUN git clone http://tfsv.tasakilab:5051/git/fukuta/h5datacreator.git && \
    source $CONDA_DIR/etc/profile.d/conda.sh && \
    conda activate bp && \
    HDF5_DIR=$CONDA_DIR/envs/bp pip install --no-build-isolation ./h5datacreator && \
    rm -rf h5datacreator

RUN pip install openpyxl numpy==1.26.4

ARG SERVER_HOSTNAME=unknown
ENV SERVER_HOSTNAME=${SERVER_HOSTNAME}


RUN wget -q https://raw.githubusercontent.com/git/git/master/contrib/completion/git-prompt.sh -O /root/.git-prompt.sh && \
    echo "source /root/.git-prompt.sh" >> /root/.bashrc && \
    echo "export PS1='\[\033[01;32m\]bpnet@docker\[\033[00m\](\[\033[01;35m\]\$SERVER_HOSTNAME\\[\\033[00m\\]):\[\033[01;34m\]\w\[\033[31m\]\$(__git_ps1 \" (%s)\")\[\033[00m\]\$ '" >> /root/.bashrc && \
    echo 'source $CONDA_DIR/etc/profile.d/conda.sh && conda activate bp' >> /root/.bashrc

COPY exts /workspace/bpnet/exts

RUN source /opt/conda/etc/profile.d/conda.sh && \
    conda activate bp && \
    export TORCH_CUDA_ARCH_LIST="6.1;8.6" && \
    cd /workspace/bpnet/exts && \
    python setup.py install
    
RUN git config --global --add safe.directory /workspace/bpnet

RUN source $CONDA_DIR/etc/profile.d/conda.sh && \
    conda activate bp && \
    pip install numpy==1.26.4 openpyxl

ENV CONDA_DEFAULT_ENV=bp
ENV PATH=/opt/conda/envs/bp/bin:$PATH

ENTRYPOINT ["/bin/bash", "-c", "source /opt/conda/etc/profile.d/conda.sh && conda activate bp && exec bash"]
