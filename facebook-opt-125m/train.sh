#!/usr/bin/env sh

set -xe

jupyter nbconvert --debug --to notebook --execute ./src/train.ipynb --inplace
