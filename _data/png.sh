#!/bin/bash

function compress()
{
  path_in=${2}/${1}


  extension="${1##*.}"
  outfile=$(basename "$1" .${extension}).png
  out="${3}/png${4}/$(dirname $1)"
  echo  "In: ${path_in} --> Out: ${out}/${outfile}"

  test -d $out || mkdir -p $out
  convert -quality ${4} "${path_in}" "${out}/${outfile}"
}
export -f compress

OUT=$2

if [ $OUT = '.' ];then
  OUT=./
fi

if [ ! -n "$1" ] || [ ! -n "$2" ] || [ ! -n "$3" ]; then
  echo "Parameter enough not supplied."
  echo "USAGE: source ./*.sh [inout] [output] [quality factor]"
  return
fi

echo "In:" $1
echo "Out:" $OUT
echo "QF:" ${3}

find $1 -type f -exec realpath --relative-to $1 {} \; |  parallel compress {} $1 $OUT $3;