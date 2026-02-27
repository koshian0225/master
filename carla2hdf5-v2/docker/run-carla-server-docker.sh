#!/bin/bash

CARLA_VERSION="0.9.14"

# docker run \
#   -t \
#   --rm \
#   hypergig/parrotsay \
#     "CARLA server @ $CARLA_VERSION"

docker run \
  -it --rm \
  -p 2000-2002:2000-2002 \
  -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
  --privileged \
  --net=host \
  -e DISPLAY=$DISPLAY\
  --gpus all \
  --name="carla-server-0.9.14" \
  carla-server:${CARLA_VERSION} \
  bash
