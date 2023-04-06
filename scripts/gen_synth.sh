#!/usr/bin/bash

BASE_OUT=../ocr-dataset/synth/
PROCESS_NUM=4
BG_NUM=600000

cd synthgenerator
echo $CWD
python generate_synth.py -v -o $BASE_OUT -w $PROCESS_NUM multistyle_template SynthForCharDiffusion synthgen_config.yaml --count $BG_NUM
