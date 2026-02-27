#!/bin/bash

IMAGE_NAME="ros:noetic-ros-core-focal"
BUILD_DIR=$(dirname $(readlink -f $0))
USER_NAME=$(whoami)

docker build \
    -t ${IMAGE_NAME} \
    --build-arg UID=$(id -u) \
    --build-arg USER_NAME=${USER_NAME} \
    --build-arg HOSTNAME=$(hostname) \
    ${BUILD_DIR}
