#!/usr/bin/env sh

set -xe

jupyter nbconvert --debug --to notebook --execute ./src/build.ipynb --output ../logs/build.ipynb
