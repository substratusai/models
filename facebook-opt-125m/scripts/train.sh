#!/usr/bin/env sh

set -xe

jupyter nbconvert --debug --to notebook --execute /model/src/train.ipynb --output /model/logs/train.ipynb
