#!/bin/bash
rm -rf waymo-od > /dev/null
git clone https://github.com/waymo-research/waymo-open-dataset.git waymo-od
cd waymo-od && git branch -a
cd waymo-od && git checkout remotes/origin/master
