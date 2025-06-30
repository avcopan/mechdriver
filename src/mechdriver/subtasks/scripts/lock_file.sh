#!/usr/bin/env bash

lock_file="${1}"
cmd=( "${@:2}" )

touch $lock_file

cleanup() {
  rm $lock_file
  echo "Deleting '$lock_file'"
}

trap cleanup EXIT

"${cmd[@]}"
