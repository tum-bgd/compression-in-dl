
conda activate vitis-ai-tensorflow2

# -------------------------------------
# Model and Dataset Parameter
# -------------------------------------

export INPUT_MODEL=${1}
export TARGET_OUT=${5}

export DATA_PATH_TRAIN=${2}
export DATA_PATH_TEST=${3}
export DATA_PATH_VALI=${4}

# network parameters
export INPUT_HEIGHT=224
export INPUT_WIDTH=224
export NET_NAME=fpgamodel
export BATCHSIZE=32

# -------------------------------------
# FPGA Deploy Configuration
# -------------------------------------

export BUILD=./build
export QUANT=${BUILD}/quantize
export COMPILE=${BUILD}/compile/
export TARGET=${BUILD}/target
export TARGET_TEMPLATE=./target_template

# Logfiles
export QUANT_LOG=quant.log
export COMP_LOG=compile.log
export TARGET_LOG=target_zcu102.log



# calibration list file
export CALIB_LIST=calib_list.json
export CALIB_IMAGES=1000

export LOG=${BUILD}/logs
mkdir -p ${LOG}

# target board
export BOARD=ZCU102
export ARCH=/opt/vitis_ai/compiler/arch/DPUCZDX8G/${BOARD}/arch.json

# DPU mode - best performance with DPU_MODE = normal
export DPU_MODE=normal
#export DPU_MODE=debug












# folders
