#!/bin/bash

CONTAINER_NAMES=(`docker ps --format "{{.Names}}"`)
NUM_CONTAINERS=${#CONTAINER_NAMES[*]}

if [ ${NUM_CONTAINERS} -eq 0 ]; then
    echo "Error: No running containers"
    exit 1
fi

echo "Specify one container"
echo "========================================"
for ((i = 0; i < ${#CONTAINER_NAMES[@]}; i++))
do
    echo "[${i}]: ${CONTAINER_NAMES[$i]}"
done
echo "========================================"
echo ""

ID=0
read -p ">>> " ID

CONTAINER_NAME=""
if expr "$ID" : "[0-9]*$" >&/dev/null; then
    CONTAINER_NAME="${CONTAINER_NAMES[$ID]}"
else
    echo "Error: Not a number"
    exit 1
fi

AS_ROOT=""
while true; do
    read -p "Attach to the container as admin (y/n): " yn
    case $yn in
        [Yy]* ) AS_ROOT="-u root"; break;;
        [Nn]* ) AS_ROOT="-u $(id -u)"; break;;
        * ) echo "Please answer y or n.";;
    esac
done

clear

docker exec -it ${AS_ROOT} ${CONTAINER_NAME} bash
