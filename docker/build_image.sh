#!/bin/bash
set -e
CONTAINER_ENGINE="docker"

if [ "$DPORO_HOME" = "" ];
then
    # Error if path to dolfinx_eqlb i not set
    echo "Path to source folder not set! Use "export DPORO_HOME=/home/.../NondimPoroelasticity_with_Growth""
    exit 1
else
    # Build docker image
    echo "DPORO_HOME is set to '$DPORO_HOME'"
    ${CONTAINER_ENGINE} pull dolfinx/dolfinx:stable
    ${CONTAINER_ENGINE} build --no-cache -f "${DPORO_HOME}/docker/Dockerfile" -t brodbeck-m/dolfinx_poro:release .
fi
