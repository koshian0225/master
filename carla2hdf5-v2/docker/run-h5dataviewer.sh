#!/bin/bash

RUN_DIR=$(dirname $(readlink -f $0))
XSOCK="/tmp/.X11-unix"
XAUTH="/tmp/.docker.xauth"

DOCKER_VOLUME="${DOCKER_VOLUME} -v $(dirname ${RUN_DIR})/shared_dir:/workspace:rw"
DOCKER_VOLUME="${DOCKER_VOLUME} -v ${XSOCK}:${XSOCK}:rw"
DOCKER_VOLUME="${DOCKER_VOLUME} -v ${XAUTH}:${XAUTH}:rw"

DOCKER_ENV="${DOCKER_ENV} -e XAUTHORITY=${XAUTH}"
DOCKER_ENV="${DOCKER_ENV} -e DISPLAY=$DISPLAY"
DOCKER_ENV="${DOCKER_ENV} -e TERM=xterm-256color"
DOCKER_ENV="${DOCKER_ENV} -e QT_X11_NO_MITSHM=1"

CARLA_VERSION="0.9.14"

docker run \
  -t \
  --rm \
  hypergig/parrotsay \
    "CARLA client @ $CARLA_VERSION"

docker run \
    -it \
    -w /workspace/script \
    --rm \
    --privileged \
    --net host \
    --gpus '"device=0"' \
    ${DOCKER_VOLUME} \
    ${DOCKER_ENV} \
    --name h5dataviewer \
    h5dataviewer:2.0
