#!/bin/bash

function compress()
{
  path_in=${2}/${1}


  extension="${1##*.}"
  outfile=$(basename "$1" .${extension}).tif
  out="${3}/tif/$(dirname $1)"
  echo  "In: ${path_in} --> Out: ${out}/${outfile}"

  test -d $out || mkdir -p $out
  convert "${path_in}" "${out}/${outfile}"
}
export -f compress

OUT=$2

if [ $OUT = '.' ];then
  OUT=./
fi

if [ ! -n "$1" ] || [ ! -n "$2" ]; then
  echo "Parameter enough not supplied."
  echo "USAGE: source ./*.sh [inout] [output]"
  return
fi

echo "In:" $1
echo "Out:" $OUT

find $1 -type f -exec realpath --relative-to $1 {} \; |  parallel compress {} $1 $OUT;