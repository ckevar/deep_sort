plotc() {
  COL="$1"
  shift 1

  PLOT_CMD="set grid"
  if [ "2" != "$COL" ]; then
    PLOT_CMD="$PLOT_CMD; set key bottom"
  fi
  PLOT_CMD="$PLOT_CMD; plot"


  for DATA in "${@}"; do

    # NOTE: If we are operating under the following file structure:
    # `experiments/dataset.d/group/logs/exp_id/results.dat`, then, the following 
    # line should be able to parse it.
    IFS="/" read _ _ _GROUP _ _EXP _ < <(echo "$DATA")
    TITLE="$_GROUP/$_EXP"
    PLOT_CMD="$PLOT_CMD '$DATA' u 1:$COL w l title '$TITLE',"
  done
  echo "GNUPLOT CMD EXECUTED"
  echo "$PLOT_CMD"
  gnuplot -p -e "$PLOT_CMD"
}

_help() {
  echo "./gather_metrics_HF.sh <task> <directory>"
  echo " "
  echo "The following tasks are available"
  echo "  - 'download-<experiment>-<dataset>'"
  echo "  - 'plot-<experiment>-<dataset>'"
  echo "  - 'blacklist-<experiment>-<dataset>'"
  echo ""
  echo "Where <experiment> is:"
  echo "  - lr: Establishing the learning rate and phase"
  echo "  - pk: Establishing the PK sampler parameters"
  echo "  - lrsmtl: learning rate, loss using softmax + triplet loss"
  echo "  - lrcls: learning rate, loss using logits (or classification) only"
  echo "  - lrufb: learning rate, loss using logits + triplet loss an fully unfreezing the backbone"
  echo "  - lrufbp: learning rate, loss using logits + triplet loss and unfreezing res mod 4 and mod 5 only and changing lr upon unfreezing"
  echo "  - seed: picking the best seed."
  echo " "
  echo "<dataset>:"
  echo "  - mot17: MOT17 dataset."
  echo "  - kitti: KITTI dataset."
  echo "  - waymov2: WaymoV2 dataset."

}



if [ -z "$2" ]; then
  printf "You need to pass a directory to operate from.\n"
  print_help
  exit 1
fi

IFS="-" read TASK EXP_ID DATASET_ID < <(echo "$1")
shift 1
if [ -z "$EXP_ID" ]; then
  printf "You need to specify an experiment name.\n"
  print_help
  exit 1
fi

if [ "plotc" = "$TASK" ]; then
  plotc "$EXP_ID" "$@"

# HELP
else
  _help
fi

