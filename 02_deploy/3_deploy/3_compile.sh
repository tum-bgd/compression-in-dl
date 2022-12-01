# compile
compile() {
  vai_c_tensorflow2  \
    --model      ${QUANT}/q_model.h5 \
    --arch       ${ARCH} \
    --output_dir ${COMPILE} \
    --net_name   ${NET_NAME} \
    --options    "{'mode':'${DPU_MODE}'}"
}

echo "-----------------------------------------"
echo "COMPILING MODEL FOR ${BOARD}.."
echo "-----------------------------------------"

rm -rf ${COMPILE}
mkdir -p ${COMPILE}
compile 2>&1 | tee ${LOG}/${COMP_LOG}

echo "-----------------------------------------"
echo "COMPILING COMPLETED"
echo "-----------------------------------------"
