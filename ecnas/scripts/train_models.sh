#!/bin/bash
#The script structure is inspired from https://stackoverflow.com/a/39376824.
########################################################################################################################
# Usage help
function usage()
{
   cat << HEREDOC

   Usage: $progname [--vertices NUM] [--edges NUM] [--epochs NUM] [--graph_file STR] [--workers NUM] [--worker_id NUM] [--worker_offset NUM] [--verbose]

   optional arguments:
     -h, --help             show this help message and exit
     -v, --vertices         max vertices
     -e, --edges            max edges
     -ep, --epochs          number of epochs to train models
     -gf, --graph_file      json graphs file for specific graphs file generated in the graphs folder
     -w, --workers          total number of workers, across all flocks
     -wid --worker_id       worker id within this flock, starting at 0
     -wo, --worker_offset   worker id offset added
     -v, --verbose          increase the verbosity of the bash script

HEREDOC
}
########################################################################################################################
# Initialize variables
progname=$(basename $0)
vertices=
edges=
epochs=
graph_file=''
workers=1
worker_id=0
worker_offset=0
estimate=0
verbose=0

########################################################################################################################
# Parse variables
OPTS=$(getopt -o "hv:e:ep:gf:w:wo:estv" --long "help,vertices:,edges:,epochs:,graph_file:,workers:,worker_id:,worker_offset:,verbose" -n "$progname" -- "$@")
if [ $? != 0 ] ; then echo "Error in command line arguments." >&2 ; usage; exit 1 ; fi
eval set -- "$OPTS"

while true; do
  case "$1" in
    -h | --help ) usage; exit; ;;
    -v | --vertices ) vertices="$2"; shift 2 ;;
    -e | --edges ) edges="$2"; shift 2 ;;
    -ep | --epochs ) epochs="$2"; shift 2 ;;
    -gf | --graph_file ) graph_file="$2"; shift 2 ;;
    -w | --workers) workers="$2"; shift 2 ;;
    -wid | --worker_id ) worker_id="$2"; shift 2 ;;
    -wo | --worker_offset ) worker_offset="$2"; shift 2 ;;
    -v | --verbose ) verbose=$((verbose + 1)); shift ;;
    -- ) shift; break ;;
    * ) break ;;
  esac
done

case $epochs in
# Change intermediate evaluations and epochs to your needs.
# NasBench-101 evaluates models at their halfway-point.
# The following values incurs evaluation after 1 epoch and max epochs for budgets in {4, 12, 36, 108}.
  4)
    intermediate_evaluations=0.5,1.0 #0.25,1.0
    ;;

  12)
    intermediate_evaluations=0.0833,1.0
    ;;

  36)
    intermediate_evaluations=0.0277,1.0
    ;;

  108)
    intermediate_evaluations=0.009259,1.0
    ;;

  *)
    echo "Error epochs have to be one of 4, 12, 36, 108." >&2
    exit 1
    ;;
esac

if (( $verbose > 0 )); then
   # Print out all parsed parameters
   cat <<EOM
   vertices=$vertices
   edges=$edges
   epochs=$epochs
   graph_file=$graph_file
   intermediate_evaluations=$intermediate_evaluations
   workers=$workers
   worker_id=$worker_id
   worker_offset=$worker_offset
   verbose=$verbose
EOM
fi

########################################################################################################################
# Run command
export TF_FORCE_GPU_ALLOW_GROWTH="true"
parentdir="${PWD%/*}"
cd $parentdir/vendors/ec_nasbench
python -m nasbench.scripts.run_evaluation --module_vertices=$vertices --max_edges=$edges --train_epochs=$epochs --models_file_name=$graph_file --intermediate_evaluations=$intermediate_evaluations --total_workers=$workers --worker_id=$worker_id --worker_id_offset=$worker_offset
