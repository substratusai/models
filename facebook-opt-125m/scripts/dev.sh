#!/usr/bin/env sh

set -xe

jupyter notebook --allow-root --ip=0.0.0.0 --NotebookApp.token='' --notebook-dir='/model'
