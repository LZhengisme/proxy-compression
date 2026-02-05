#!/bin/bash
set -e

# Input arguments
src_dir="$1"          # Source directory
glob_pattern="$2"     # Glob pattern to match files (e.g. *.txt)
nchunks="$3"          # Number of chunks
out_dir="$4"          # Output directory
dataset="$5"          # dataset name

mkdir -p "$out_dir"

{
    # Use find to get all matching files and cat them all
    find "$src_dir" -name "$glob_pattern" -exec cat {} \;
} | shuf | split -n r/${nchunks} --numeric-suffixes=1 --suffix-length=${#nchunks} --additional-suffix=.jsonl - "${out_dir}/${dataset}.chunk."
