#!/bin/bash


echo "download mnist data"

if [ ! -d "data" ]; then
  mkdir data
fi

urls=(
  https://storage.googleapis.com/cvdf-datasets/mnist/train-images-idx3-ubyte.gz
  https://storage.googleapis.com/cvdf-datasets/mnist/train-labels-idx1-ubyte.gz
  https://storage.googleapis.com/cvdf-datasets/mnist/t10k-images-idx3-ubyte.gz
  https://storage.googleapis.com/cvdf-datasets/mnist/t10k-labels-idx1-ubyte.gz)

for url in "${urls[@]}"; do
  filename=$(basename "$url")
  curl -o "data/$filename" "$url"
  gunzip "data/$filename"
done

echo "Done"

