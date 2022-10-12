#!/bin/bash

cd ..
dir="$PWD/vendors/ec_nasbench/nasbench/data/train_model_results"
echo "Removing .ckpt files.."
before="$(df -h | grep /dev/nvme0n1p2)"
#find $dir -type f -name model.ckpt\* -exec rm {} \;
find $dir -type d -name eval_\* -prune -exec rmdir {} \;
#find $dir -type f -name events.out.tfevents.\* -exec rm {} \;
#find $dir -type f -name checkpoint\* -exec rm {} \;
#find $dir -type f -name graph.pbtxt\* -exec rm {} \;

after="$(df -h | grep /dev/nvme0n1p2)"
echo " | Before: $before"
echo " | After:  $after"
echo "Done!"
printf %"$COLUMNS"s | tr " " "-"
