#!/bin/bash
#The script structure is inspired from https://stackoverflow.com/a/39376824.
########################################################################################################################
# Usage help
function usage()
{
   cat << HEREDOC

   Usage: $progname [--vertices NUM] [--edges NUM] [--print_examples NUM] [--energy] [--verbose]

   optional arguments:
     -h, --help             show this help message and exit
     -v, --vertices         max vertices
     -e, --edges            max edges
     -p, --print_examples   number of examples to print from the generated dataset
     -ene, --energy           option to generate dataset including energy metrics
     -v, --verbose          increase the verbosity of the bash script

HEREDOC
}
########################################################################################################################
# Initialize variables
progname=$(basename $0)
vertices=
edges=
print_examples=1
energy=false
verbose=0

########################################################################################################################
# Parse variables
OPTS=$(getopt -o "hv:e:p:cv" --long "help,vertices:,edges:,print_examples:,energy,verbose" -n "$progname" -- "$@")
if [ $? != 0 ] ; then echo "Error in command line arguments." >&2 ; usage; exit 1 ; fi
eval set -- "$OPTS"

while true; do
  case "$1" in
    -h | --help ) usage; exit; ;;
    -v | --vertices ) vertices="$2"; shift 2 ;;
    -e | --edges ) edges="$2"; shift 2 ;;
    -p | --print_examples ) print_examples="$2"; shift 2 ;;
    -ene | --energy ) carbon=true; shift ;;
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
   print_examples=$print_examples
   energy=energy
   verbose=$verbose
EOM
fi

########################################################################################################################
# Run command
cd ../vendors/ec_nasbench
if (( $energy == true)); then
  echo "Energy"
  python -m nasbench.scripts.generate_dataset_energy --module_vertices=$vertices --max_edges=$edges --examples=$print_examples
else
  python -m nasbench.scripts.generate_dataset --module_vertices=$vertices --max_edges=$edges --examples=$print_examples
fi
