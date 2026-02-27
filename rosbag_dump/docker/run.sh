#!/bin/bash
clear
docker run -it --rm otherguy/pokemonsay "rosbag_dump"

set -e

AUTOWARE_HOST_DIR=""

SUFFIX=""
RUNTIME=""

XSOCK=/tmp/.X11-unix
XAUTH=$HOME/.Xauthority

SHARED_DOCKER_DIR=/workspace
SHARED_HOST_DIR=$HOME/rosbag_dump/shared_dir/rosbag_dump

VOLUMES="--volume=$XSOCK:$XSOCK:rw
         --volume=$XAUTH:$XAUTH:rw
         --volume=$SHARED_HOST_DIR:$SHARED_DOCKER_DIR:rw"
xhost +

docker run \
    -it -w /workspace/ --rm --name rosbag_dump --shm-size=10gb \
    $VOLUMES \
    --env="XAUTHORITY=${XAUTH}" \
    --env="DISPLAY=${DISPLAY}" \
    --gpus=all \
    --net=host \
    --env HOME=/workspace \
    ros:noetic-ros-core-focal-v2 bash
