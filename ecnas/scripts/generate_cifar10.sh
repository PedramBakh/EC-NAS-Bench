#!/bin/bash

cd ..
old_pwd=$(pwd)
dir="$old_pwd/vendors/ec_nasbench/nasbench/data/cifar10-tfrecords"
file="$old_pwd/vendors/ec_nasbench/nasbench/scripts/generate_cifar10_tfrecords.py --data_dir=$dir"
if [ -d $dir ]
  then
    echo "Directory '$dir' already exists.."
    echo "Generating CIFAR-=10 records.."
    python $file
    printf %"$COLUMNS"s | tr " " "-"
  else
    echo "Creating directory for CIFAR-10 records..."
    mkdir $dir
    echo "Generating CIFAR-=10 records.."
    python $file
    printf %"$COLUMNS"s | tr " " "-"
fi