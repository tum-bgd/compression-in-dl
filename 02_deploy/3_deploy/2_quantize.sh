# quantize
quantize() {
    echo "Making calibration images.."  
    
    python tf_gen_images.py  \
        --image_dir    ${QUANT}/images \
        --calib_list   ${CALIB_LIST} \
        --max_images   ${CALIB_IMAGES} \
        --data_path    ${DATA_PATH_TRAIN} \
        --input_height ${INPUT_HEIGHT} \
        --input_width  ${INPUT_WIDTH}

    python quantize.py \
            --model     ${INPUT_MODEL} \
            --q_model   ${QUANT} \
            --batchsize ${BATCHSIZE} \
            --datapath  ${QUANT}/${CALIB_LIST} \
            --eval_data ${DATA_PATH_TEST} \
            --verbose \
            --evaluate
}

echo "-----------------------------------------"
echo "QUANTIZE KERAS MODEL"
echo "-----------------------------------------"
rm -rf ${QUANT} 
mkdir -p ${QUANT}/images
quantize 2>&1 | tee ${LOG}/${QUANT_LOG}
rm -rf ${QUANT}/images
echo "-----------------------------------------"
echo "QUANTIZATION COMPLETED"
echo "-----------------------------------------"

