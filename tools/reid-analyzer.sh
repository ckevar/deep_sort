__fetch_data() {
  _REMOTE_DIR="$1"
  _NAME="$2"
  _LOCAL_DIR="$3"
  _DAT_FILE="$4"

  if [ ! -d "$_LOCAL_DIR" ]; then
    URL="https://huggingface.co/ckevar/wrn-deepSORT/resolve/main/$_REMOTE_DIR/logs/$_NAME/$_DAT_FILE"

    echo "Attempt to download $URL"

    mkdir -p "$_LOCAL_DIR"

    wget -q "$URL" -P "$_LOCAL_DIR"

    if [ ! -f "$_LOCAL_DIR/$_DAT_FILE" ]; then
      echo "  -> Not found remotely: $_NAME/$_DAT_FILE."
      rm -r "$_LOCAL_DIR"
    else 
      echo "  -> Sucessfully downloaded $_NAME/$_DAT_FILE."
    fi

  else
    echo "Found locally: $_NAME/$_DAT_FILE."
  fi

}

__fetch_cache() {
  _REMOTE_DIR="$1"
  _LOCAL_DIR="$2"

  URL="https://huggingface.co/ckevar/wrn-deepSORT/resolve/main/$_REMOTE_DIR/experiments.txt"
  mkdir -p "$_LOCAL_DIR"
  wget -q -O "$_LOCAL_DIR/experiments.txt" "$URL"

  if [ ! -f "$_LOCAL_DIR/experiments.txt" ]; then
    echo "NOT FOUND: Unable to fetch $_REMOTE_DIR from $_URL"
    exit 1
  fi
}

wget_dat () {
  DATASET_ID="$1"
  GROUP_ID="$2"
  TARGET_DIR="$3"
  DATASET_ROOT="$TARGET_DIR/$DATASET_ID"

  TARGET_DIR="$DATASET_ROOT.d/$GROUP_ID"

  for REMOTE_DIR in $(yq ".$GROUP_ID.experiments.*.result_path" "$DATASET_ROOT.yaml"); do
    URL="https://huggingface.co/ckevar/wrn-deepSORT/resolve/main/$REMOTE_DIR/$_DAT_FILE"
    wget -qO "$URL"
  done

}

download_data_margin_not() {
  DATASET_ID="$1"
  TARGET_DIR="$2"

  REMOTE_DIR="margin_not-$DATASET_ID"
  LOCAL_DIR="$TARGET_DIR/$REMOTE_DIR"
  __fetch_cache "$REMOTE_DIR" "$LOCAL_DIR"


  EXPERIMENT_DAT_FILE="results_lr0.001_p22_k27.dat"
  while read EXPERIMENT_NAME; do
    EXPERIMENT_LOCAL_DIR="$LOCAL_DIR/logs/$EXPERIMENT_NAME"
    __fetch_data "$REMOTE_DIR" "$EXPERIMENT_NAME" "$EXPERIMENT_LOCAL_DIR" "$EXPERIMENT_DAT_FILE"
  done < "$LOCAL_DIR/experiments.txt"

}

download_data_lr_not() {
  DATASET_ID="$1"
  TARGET_DIR="$2"
  REMOTE_DIR_BASE="lr_not-$DATASET_ID-at-phase"

  declare -a phases=("3" "6")

  for ph in "${phases[@]}"; do
    REMOTE_DIR="$REMOTE_DIR_BASE${ph}"
    LOCAL_DIR="$TARGET_DIR/$REMOTE_DIR"
    __fetch_cache "$REMOTE_DIR" "$LOCAL_DIR"

    while read EXPERIMENT_NAME; do
      EXPERIMENT_DAT_FILE="results_${EXPERIMENT_NAME}_p22_k27.dat"
      EXPERIMENT_LOCAL_DIR="$LOCAL_DIR/logs/$EXPERIMENT_NAME"
      __fetch_data "$REMOTE_DIR" "$EXPERIMENT_NAME" "$EXPERIMENT_LOCAL_DIR" "$EXPERIMENT_DAT_FILE"
    done < "$LOCAL_DIR/experiments.txt"
  done

}

download_data_seed() {
  DATASET_ID="$1"
  TARGET_DIR="$2"
  
  # Fetch the experiments file from the main directory in repo
  REMOTE_DIR="seed-$DATASET_ID"
  LOCAL_DIR="$TARGET_DIR/$REMOTE_DIR"
  __fetch_cache "$REMOTE_DIR" "$LOCAL_DIR"
  
  EXPERIMENT_DAT_FILE="results_lr0.0001_p22_k27.dat"
  while read EXPERIMENT_NAME; do
    EXPERIMENT_LOCAL_DIR="$LOCAL_DIR/logs/$EXPERIMENT_NAME"
    __fetch_data "$REMOTE_DIR" "$EXPERIMENT_NAME" "$EXPERIMENT_LOCAL_DIR" "$EXPERIMENT_DAT_FILE"
  done < "$LOCAL_DIR/experiments.txt"

}

download_data_lrufbp() {
DATASET_ID="$1"
TARGET_DIR="$2"

declare -a lrs=("0.001" "0.0001")
declare -a unfreeze_at_epoch=("10")

for lr in "${lrs[@]}"; do
  for uae in "${unfreeze_at_epoch[@]}"; do
    EXPERIMENT_DIR="lrufbp-${DATASET_ID}-at-${uae}"
    EXPERIMENT_NAME="lrtpo${lr}"
    EXPERIMENT_DAT_FILE="results_lr${lr}_p22_k27.dat"
    EXPERIMENT_LOCAL_DIR="$TARGET_DIR/$EXPERIMENT_DIR/logs/$EXPERIMENT_NAME"

    __fetch_data \
      "$EXPERIMENT_DIR" \
      "$EXPERIMENT_NAME" \
      "$EXPERIMENT_LOCAL_DIR" \
      "$EXPERIMENT_DAT_FILE"

  done
done
}

download_data_lrufb() {
DATASET_ID="$1"
TARGET_DIR="$2"

declare -a lrs=("0.001" "0.0002")
declare -a unfreeze_at_epoch=("14" "35" "50" "75")

for lr in "${lrs[@]}"; do
  for uae in "${unfreeze_at_epoch[@]}"; do
    EXPERIMENT_DIR="lrufbp-${DATASET_ID}-at-${uae}"
    EXPERIMENT_NAME="lr${lr}"
    EXPERIMENT_DAT_FILE="results_lr${lr}_p22_k27.dat"
    EXPERIMENT_LOCAL_DIR="$TARGET_DIR/$EXPERIMENT_DIR/logs/$EXPERIMENT_NAME"

    __fetch_data \
      "$EXPERIMENT_DIR" \
      "$EXPERIMENT_NAME" \
      "$EXPERIMENT_LOCAL_DIR" \
      "$EXPERIMENT_DAT_FILE"

  done
done
}

download_data_lrcls() {
DATASET_ID="$1"
TARGET_DIR="$2"

declare -a lrs=("0.0001" "0.001" "0.01")
declare -a phases=("6")

for lr in "${lrs[@]}"; do
  for ph in "${phases[@]}"; do
    EXPERIMENT_DIR="lrcls-${DATASET_ID}-at-phase${ph}"
    EXPERIMENT_NAME="lr${lr}"
    EXPERIMENT_DAT_FILE="results_lr${lr}_p22_k27.dat"
    EXPERIMENT_LOCAL_DIR="$TARGET_DIR/$EXPERIMENT_DIR/logs/$EXPERIMENT_NAME"

    __fetch_data \
      "$EXPERIMENT_DIR" \
      "$EXPERIMENT_NAME" \
      "$EXPERIMENT_LOCAL_DIR" \
      "$EXPERIMENT_DAT_FILE"

  done
done
}
download_data_lrsmtl() {
DATASET_ID="$1"
TARGET_DIR="$2"

declare -a lrs=("1e-06" "0.0001" "0.001" "0.01")
declare -a phases=("1" "3" "6")

for lr in "${lrs[@]}"; do
  for ph in "${phases[@]}"; do
    EXPERIMENT_DIR="lrsmtl-${DATASET_ID}-at-phase${ph}"
    EXPERIMENT_NAME="lr${lr}"
    EXPERIMENT_DAT_FILE="results_lr${lr}_p22_k27.dat"
    EXPERIMENT_LOCAL_DIR="$TARGET_DIR/$EXPERIMENT_DIR/logs/$EXPERIMENT_NAME"

    __fetch_data \
      "$EXPERIMENT_DIR" \
      "$EXPERIMENT_NAME" \
      "$EXPERIMENT_LOCAL_DIR" \
      "$EXPERIMENT_DAT_FILE"

  done
done
}


download_data_pk() {
DATASET_ID="$1"
TARGET_DIR="$2"
LR="${3:-0.0002}"
PHASE="${4:-6}"

declare -a p_values=("2" "4" "8" "16" "22")
declare -a k_values=("2" "4" "8" "16" "27")

for p in "${p_values[@]}"; do
  for k in "${k_values[@]}"; do
    EXPERIMENT_DIR="pk-${DATASET_ID}-at-phase${PHASE}"
    EXPERIMENT_NAME="p${p}-k${k}"
    EXPERIMENT_DAT_FILE="results_lr${LR}_p${p}_k${k}.dat"
    EXPERIMENT_LOCAL_DIR="$TARGET_DIR/$EXPERIMENT_DIR/logs/$EXPERIMENT_NAME"

    __fetch_data \
      "$EXPERIMENT_DIR" \
      "$EXPERIMENT_NAME" \
      "$EXPERIMENT_LOCAL_DIR" \
      "$EXPERIMENT_DAT_FILE"

  done
done

}

download_data_lr() {
  DATASET="$1"
  TARGET_DIR="$2"

  declare -a lrs=("1e-06" "0.0001" "0.0002" "0.0003" "0.0004" "0.0005" "0.001" "0.0015" "0.002" "0.0029" "0.003")
  declare -a phases=("1" "2" "3" "4" "5" "6")


  for lr in "${lrs[@]}"; do
    for ph in "${phases[@]}"; do
      EXPERIMENT_DIR="lr-${DATASET}-at-phase${ph}"
      EXPERIMENT_NAME="lr${lr}"
      EXPERIMENT_DAT_FILE="results_lr${lr}_p22_k27.dat"
      EXPERIMENT_LOCAL_DIR="$TARGET_DIR/$EXPERIMENT_DIR/logs/$EXPERIMENT_NAME"

      __fetch_data \
        "$EXPERIMENT_DIR" \
        "$EXPERIMENT_NAME" \
        "$EXPERIMENT_LOCAL_DIR" \
        "$EXPERIMENT_DAT_FILE"

    done
  done
}


plot() {
  DATASET_ID="$1"
  EXP_ID="$2"
  TARGET_DIR="$3"
  Y_AXIS="${4:-3}"

  BLACKLIST="blacklist-$EXP_ID-$DATASET_ID.txt"
  if [ ! -f "$BLACKLIST" ]; then
    touch "$BLACKLIST"
  fi

  #PLOT_CMD="set key outside; plot"
  #PLOT_CMD="plot"
  PLOT_CMD="set key bottom; plot"
  #PLOT_CMD="unset key; plot"
  for DATA in $TARGET_DIR/$EXP_ID-*/logs/*/*.dat; do

    PARENT_DIR=$(dirname "$DATA")
    echo "DATA FILE $DATA"

    if grep -q "$PARENT_DIR" "$BLACKLIST"; then
      continue
    fi

    PARENT_DIR=$(basename $(dirname $(dirname "$PARENT_DIR")))

    #TAG="$f"
    TAG=$(basename "$DATA")
    TAG="$PARENT_DIR/$TAG"

    PLOT_CMD="$PLOT_CMD '$DATA' u 1:$Y_AXIS w l title '$TAG',"
  done
  gnuplot -p -e "$PLOT_CMD"
}

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
  echo "GNU SCRIPT EXECUTED"
  echo "$PLOT_CMD"
  gnuplot -p -e "$PLOT_CMD"
}

blacklist() {
  DATASET_ID="$1"
  EXP_ID="$2"
 
  BLACKLIST="blacklist-$EXP_ID-$DATASET_ID.txt"
  if [ ! -f "$BLACKLIST" ]; then
    touch "$BLACKLIST"
  fi

  if [ -z "$3" ]; then
    cat "$BLACKLIST"
    exit 0
  fi

  shift 2

  for ignored in "${@}"; do
    if grep -q "$ignored" "$BLACKLIST"; then
      continue
    fi

    echo "$ignored" >> "$BLACKLIST"

  done
 
}

print_help() {
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

DATASET_ID="${DATASET_ID:-mot17}"


# DOWNLOADS
if [ "download" = "$TASK" ]; then
  "download_data_$EXP_ID" "$DATASET_ID" $1

elif [ "wget" = "$TASK" ]; then
  wget_dat "$DATASET_ID" "$EXP_ID" $1
  
# PLOTTINGS
elif [ "plot" = "$TASK" ]; then
  plot "$DATASET_ID" "$EXP_ID" "$@"

elif [ "plotc" = "$TASK" ]; then
  plotc "$EXP_ID" "$@"

# BLACK LIST: This are experiments that are not plotten for every category
elif [ "blacklist" = "$TASK" ]; then
  blacklist "$DATASET_ID" "$EXP_ID" "$@"

# HELP
else
  print_help
fi

