export OPENBLAS_NUM_THREADS=64
export GOTO_NUM_THREADS=64
export OMP_NUM_THREADS=64


if [ ! -n "$2" ]; then
  echo "Parameter enough not supplied."
  echo "USAGE: source ./*.sh [dataset name] [path to parent location]"
  return
fi

d_path=$2
dataset=$1

echo "------------------------------------------------"
echo "Start with ${dataset} (location_ ${d_path})"
echo "------------------------------------------------"

folder=data_split
if [ "$dataset" == "EuroSAT" ]; then
   folder=data_truecolor_split
fi

python3 quantize.py -i ${d_path}/${dataset}/${folder} -o ${dataset}


rm -rf ${dataset}/*.csv

file=$(find $dataset/reference/km-8/val_0 -type f  | head -n 1)
extension="${file##*.}"

echo $extension

if [ "$extension" == "TIFF" ] && [ "$extension" == "tiff" ] && [ "$extension" == "TIF" ] && [ "$extension" == "tif" ]; then
    source ./tiff.sh $1/reference $1/
fi

if [ "$extension" == "PNG" ] && [ "$extension" == "png" ]; then
    source ./png.sh $1/reference $1/ 75
    source ./png.sh $1/reference $1/ 95
fi

if [ "$extension" == "BMP" ] && [ "$extension" == "bmp" ]; then
    source ./bmp.sh $1/reference $1/
fi

source ./jpg.sh $1/reference $1/ 75
source ./jpg.sh $1/reference $1/ 50
source ./jpg.sh $1/reference $1/ 25
source ./jpg.sh $1/reference $1/ 10
source ./jpg.sh $1/reference $1/ 5
source ./jpg.sh $1/reference $1/ 1