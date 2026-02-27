#!/bin/bash
RUN_DIR=$(dirname $(readlink -f $0))
PKG_DIR=$(dirname ${RUN_DIR})

docker build \
  -f ${RUN_DIR}/Dockerfile \
  -t "fukutasakilab/bpnet:first" \
  --build-arg SERVER_HOSTNAME=$(hostname) \
  ${PKG_DIR}