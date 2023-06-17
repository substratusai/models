#!/usr/bin/env sh

set -xe

jupyter nbconvert --debug --to notebook --execute /model/src/build.ipynb --output /model/logs/build.ipynb
