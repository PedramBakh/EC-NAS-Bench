#!/bin/bash
#The script structure is inspired from https://stackoverflow.com/a/39376824.
########################################################################################################################
# Usage help
function usage()
{
   cat << HEREDOC

   Usage: $progname [--vertices NUM] [--edges NUM] [--samples NUM] [--verbose]

   optional arguments:
     -h, --help           show this help message and exit
     -v, --vertices       max vertices
     -e, --edges          max edges
     -s, --samples        number of graphs to sample
     -v, --verbose        increase the verbosity of the bash script

HEREDOC
}
########################################################################################################################
# Initialize variables
progname=$(basename $0)
vertices=
edges=
samples='-1'
verbose=0

########################################################################################################################
# Parse variables
OPTS=$(getopt -o "hv:e:s:v" --long "help,vertices:,edges:,samples:,verbose" -n "$progname" -- "$@")
if [ $? != 0 ] ; then echo "Error in command line arguments." >&2 ; usage; exit 1 ; fi
eval set -- "$OPTS"

while true; do
  case "$1" in
    -h | --help ) usage; exit; ;;
    -v | --vertices ) vertices="$2"; shift 2 ;;
    -e | --edges ) edges="$2"; shift 2 ;;
    -s | --samples ) samples="$2"; shift 2 ;;
    -v | --verbose ) verbose=$((verbose + 1)); shift ;;
    -- ) shift; break ;;
    * ) break ;;
  esac
done

if (( $verbose > 0 )); then
   # Print out all parsed parameters
   cat <<EOM
   vertices=$vertices
   edges=$edges
   samples=$samples
   verbose=$verbose
EOM
fi

########################################################################################################################
# Run command
cd ../vendors/ec_nasbench
python -m nasbench.scripts.generate_graphs --max_vertices=$vertices --max_edges=$edges --num_samples=$samples