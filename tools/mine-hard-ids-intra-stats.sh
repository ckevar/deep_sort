INTRA_FILE="$1"

OUT_DIR="$2"
SRC_MAP="$3"

if [ -n "$2" ]; then
    if [ -d "$2" ]; then
        OUT_DIR="$2"
    elif [ -f "$2" ]; then
        SRC_MAP="$2"
    fi
fi


INTRA_FOI="/tmp/mine-hard-ids-inter-stats.tmp"

EXP_NAME=$(basename "$INTRA_FILE")
EXP_NAME="${EXP_NAME%.*}"



if [ -n "$OUT_DIR" ]; then
    OUT_DIR=$(dirname "$INTRA_FILE")
fi

OUT_FILE="$OUT_DIR/$EXP_NAME"

grep -v "#" "$INTRA_FILE" > "$INTRA_FOI"

echo 
echo " 1. Number of Patch Hard Positives":
wc -l "$INTRA_FOI"

echo 
echo " 2. Number of Hard Positives:"
sort -nk1,1 "$INTRA_FOI" | awk '{print $1}' | uniq -c | sort -nrk1,1 \
    > "$OUT_FILE-hard_positives-independent_sparsity.txt"
wc -l "$OUT_FILE-hard_positives-independent_sparsity.txt"

echo ""
echo " 3. Number of Hard Negatives:"
sort -nuk2,2 "$INTRA_FOI" | awk '{print $2}' | uniq -c | sort -nrk1,1 \
    > "$OUT_FILE-hard_negatives-attractors.txt"
wc -l "$OUT_FILE-hard_negatives-attractors.txt"

sort -nk1,1 "$INTRA_FOI" | awk '{print $1" "$2}' | uniq -c | sort -nrk1,1 > "$OUT_FILE-joint_sparsity.txt"


if [ -n "$SRC_MAP" ]; then
    # TODO: 
    # find in the map the lines and generate a file where the hard positive image patches are.
    awk 'FNR=NR {lines[$4]; next} (FNR in lines)' "$INTRA_FOI" "$SRC_MAP"
fi

echo 
echo " 4. Joint Sparsity saved int $OUT_FILE-joint_sparsity.txt"
