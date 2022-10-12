#!/bin/bash
cd ..
path="$PWD/vendors/ec_nasbench/nasbench/data/train_model_results/energy/*"

echo "Counting # of trained models.."
for dir in $path
do
  printf "$(basename $dir):\n";
  for folder in $(ls $dir | sort -g)
    do
      printf "    | $(basename $folder): "; find "$dir/$folder/" -type f -name 'results.json' -exec echo . \; | wc -l ;
    done
done
echo "Done!"
printf %"$COLUMNS"s | tr " " "-"
