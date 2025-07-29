#!/bin/bash

# Install Git LFS and pull large files
apt-get update
apt-get install git-lfs -y
git lfs install
git lfs pull
