#!/bin/bash
RUN_DIR=$(dirname $(readlink -f $0))
PKG_DIR=$(dirname ${RUN_DIR})
SRC_IMAGE="pytorch/pytorch:1.7.0-cuda11.0-cudnn8-devel"
DST_IMAGE="fukutasakilab/pmod_bpre:latest"

docker build \
  -t "${DST_IMAGE}" \
  --build-arg SRC_IMAGE="${SRC_IMAGE}" \
  ${RUN_DIR}
