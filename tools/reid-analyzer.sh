plotc() {
  local COL="$1"
  shift 1

  PLOT_CMD="set grid"
  if [ "2" != "$COL" ]; then
    PLOT_CMD="$PLOT_CMD; set key bottom"
  else
    #PLOT_CMD="$PLOT_CMD; set logscale x"
    PLOT_CMD="$PLOT_CMD;"
  fi

  PLOT_CMD="$PLOT_CMD; plot"
  MAX_PEAK_VAL="0.0"


  for DATA in "${@}"; do

    if [ ! -f "$DATA" ]; then
      echo "Cannot find file '$DATA'"
      continue
    fi

    # NOTE: If we are operating under the following file structure:
    # `experiments/dataset.d/group/logs/exp_id/results.dat`, then, the following 
    # line should be able to parse it.
    IFS="/" read _ _ _GROUP _ _EXP _ < <(echo "$DATA")
    TITLE="$_GROUP/$_EXP"
    PLOT_CMD="$PLOT_CMD '$DATA' u 1:$COL w l title '$TITLE',"


    MAX_PEAK_VAL=$(awk -vMAX="$MAX_PEAK_VAL" 'NR > 1 && $3 > MAX {MAX = $3} END {print MAX}' "$DATA")

  done
  
  echo ""
  echo "PEAK mAP: $MAX_PEAK_VAL"
  echo "GNUPLOT CMD EXECUTED"
  echo "$PLOT_CMD"
  gnuplot -p -e "$PLOT_CMD"
}

_help() {
  echo "Usage: $0 <task> [arguments]"
  echo ""
  echo "task:"
  echo "  - 'plotc-<column> <*.dat files with metrics>'"
  echo "  - 'download-<experiment>-<dataset>' not implemented"
  echo ""

}

download_metrics () {
  local DATASET="$1"
  local EXP_NAME="$2"
  local ROOT_PATH="${3:-default}"
  local EXP_ID="$4"

  ROOT_PATH="$ROOT_PATH/$DATASET.d/$EXP_ID-$EXP_NAME"
  BASE_URL="https://huggingface.co/ckevar/wrn-deepSORT/resolve/main/$DATASET-$EXP_NAME"

  mkdir -p "$ROOT_PATH"
  wget -q "$BASE_URL/experiments.txt" -O "$ROOT_PATH/experiments.txt"
  FETCH_FLAG=1

  while read RUN_NAME; do
    RESULT_PATH="$ROOT_PATH/logs/$RUN_NAME"

    if [ ! -f "$RESULT_PATH/results.dat" ]; then
      echo "Downloading experiment name $RUN_NAME..."
      mkdir -p "$RESULT_PATH"
      wget -q "$BASE_URL/logs/$RUN_NAME/results.dat" -O "$RESULT_PATH/results.dat"
      FETCH_FLAG=0
    fi

  done < "$ROOT_PATH/experiments.txt"

  if [ $FETCH_FLAG -eq 1 ]; then
    echo ""
    echo "All fetched nothing to do."
  else
    echo "All done."
    echo ""
  fi

}


if [ -z "$2" ]; then
  printf "You need to pass a directory to operate from.\n"
  print_help
  exit 1
fi

IFS="-" read TASK ARG0 ARG1 < <(echo "$1")
shift 1
if [ -z "$ARG0" ]; then
  printf "You need to specify an experiment name or column to draw\n"
  print_help
  exit 1
fi

case $TASK in
  "plotc")
    plotc "$ARG0" "$@"
    ;;
  
  "fetch")
    download_metrics $ARG0 $ARG1 "$@"
    ;;

  "help"|"-h"|"--help")
    _help
    ;;
  *)
    _help
    if [ -n $TASK ]; then
      echo ""
      echo "Error: Unknow task $TASK"
      exit 1
    fi
    ;;
esac

