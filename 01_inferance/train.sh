if [[ "$(docker images -q rsmodels/tf_train 2> /dev/null)" == "" ]]; then
  cd container
  make
  cd ..
fi

yourfilenames=`ls $1`



mkdir -p log

for eachfile in $yourfilenames
do
    extension="${eachfile##*.}"
    runid="$(basename ${eachfile} .${extension})"
    runid=(${runid//_/ })
    runid=${runid[0]}


    echo "START:$(date)" | tee -a log/logfile.${runid}
    echo "PWD:$(pwd)" | tee -a log/logfile.${runid}
    echo "HOST:$(hostname)"| tee -a log/logfile.${runid}

    docker run --rm -it --gpus all --user=$(id -u):$(id -g) -v $PWD/../../:/tf -w /tf/compressionindeeplearning/01_inferance --env HDF5_USE_FILE_LOCKING=FALSE rsmodels/tf_train /bin/bash -c "source ./set_env.sh; python3 entrypoint.py $1/${eachfile} 2>&1 | tee -a log/logfile.${runid}"
    #CONTAINER_ID=$(docker run --rm -it --gpus all --user=$(id -u):$(id -g) -v $PWD/../:/tf -w /tf/rsModels_v2 --env HDF5_USE_FILE_LOCKING=FALSE rsmodels/tf_train /bin/bash -c "source ./set_env.sh; python3 entrypoint.py ${eachfile} 2>&1")
    #echo "Container ID: ${CONTAINER_ID}"
    #docker logs $CONTAINER_ID --follow
    echo "END:$(date)" | tee -a log/logfile.${runid}
    echo "=============================="
done
