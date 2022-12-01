echo "-----------------------------------------"
echo "EVALUATING THE INPUT GRAPH.."
echo "-----------------------------------------"
    python evaluate_graph.py \
        --model ${INPUT_MODEL} \
        --batchsize ${BATCHSIZE} \
        --datapath ${DATA_PATH_TEST}
echo "-----------------------------------------"
echo "EVALUATION COMPLETED"
echo "-----------------------------------------"