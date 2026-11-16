#!/bin/bash

# Create the new main directory
mkdir -p z0.2

# Create the grid directories in z0.2
for i in {0..3}; do
    mkdir -p z0.2/grid$i
done

# Copy specific files from each grid folder
for i in {0..3}; do
    cp z0.1/grid$i/input.i3d z0.2/grid$i/
    cp z0.1/grid$i/run_xcompact3d.sh z0.2/grid$i/
done
