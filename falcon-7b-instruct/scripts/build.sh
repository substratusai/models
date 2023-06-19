#!/usr/bin/env sh

set -xe

export HF_HUB_DISABLE_PROGRESS_BARS=1
jupyter nbconvert --debug --to notebook --execute /model/src/build.ipynb --output /model/logs/build.ipynb
