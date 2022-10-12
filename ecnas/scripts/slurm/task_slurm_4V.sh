#!/bin/bash
#SBATCH --job-name=ec_4V

#normal cpu stuff: allocate cpus, memory
#SBATCH --ntasks=1

#SBATCH --exclude=a00610,a00621,a00636,a00637,a00701,a00818,a00861,a00862,a00863,a00885,a00886,a00756,a00757,a00860,a00701
#SBATCH --cpus-per-task=8 --mem=8000M

# we run on the gpu partition and we allocate 1 gpu
#SBATCH -p gpu --gres=gpu:1

#We expect that our program should not run longer than 10 hours
#Note that a program will be killed once it exceeds this time!
#SBATCH --time=10:00:00

#SBATCH --array=0-7

#your script, in this case: write the hostname and the ids of the chosen gpus.
hostname
echo $CUDA_VISIBLE_DEVICES

wid=$((SLURM_ARRAY_TASK_ID))

# Add delay between new jobs to possible prevent API restrictions.
sleep $((30))

# worker_offset is the number of workers in all flocks prior to this flock, e.g., if there is another flock with 2 workers, then this flock needs offset 2).
# This is Flock 1 with 8 workers.
parentdir="${PWD%/*}"
$parentdir/scripts/train_models.sh --vertices=4 --edges=9 --epochs=108 --graph_file="4V9E_all_91" --workers=8 --worker_id=$wid --worker_offset=0 --verbose
