echo "-----------------------------------------"
echo "MAKE TARGET STARTED.."
echo "-----------------------------------------"
mkdir -p ${TARGET}/images

cp ${QUANT}/classes.txt ${TARGET}
echo "  Copied Classnames"



python tf_gen_images.py  \
    --image_dir    ${TARGET}/images \
    --max_images   ${CALIB_IMAGES} \
    --data_path    ${DATA_PATH_VALI} \
    --input_height ${INPUT_HEIGHT} \
    --input_width  ${INPUT_WIDTH} \
    --calib_list   ${CALIB_LIST}

python -u target.py \
    -m  ${COMPILE}/${NET_NAME}.xmodel \
    -a  ${TARGET_TEMPLATE}/ \
    -i  ${TARGET} \
    -ih ${INPUT_HEIGHT} \
    -iw ${INPUT_WIDTH} \
    -t  ${BUILD}/${TARGET_OUT} \
    -n  ${CALIB_IMAGES} \
    2>&1 | tee ${LOG}/${TARGET_LOG}

mkdir -p targets_zcu102
cp -r ${BUILD}/${TARGET_OUT} ./targets_zcu102/
rm -rf ./build/
echo "-----------------------------------------"
echo "MAKE TARGET COMPLETED"
echo "-----------------------------------------"
