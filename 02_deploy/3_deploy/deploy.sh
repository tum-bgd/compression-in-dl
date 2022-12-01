#!/bin/bash

if [ ! -n "$1" ] || [ ! -n "$2" ] || [ ! -n "$3" ]; then
  echo "Parameter enough not supplied."
  echo "USAGE: source ./*.sh [Dataset Path] [Model Path] [Target Name]"
  return
fi

MODEL=${2}
TARGET_OUT=${3}
TRAIN=${1}/train
TEST=${1}/val_0
VALI=${1}/val_1

source ./0_setenv.sh $MODEL $TEST $TRAIN $VALI $TARGET_OUT
source ./1_eval_input_graph.sh
source ./2_quantize.sh
source ./3_compile.sh
source ./4_make_target.sh
