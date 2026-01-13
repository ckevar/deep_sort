INTRA_FILE="$1"
OUT_DIR="$2"
SRC_MAP="$3"

if [ ! -f "$1" ]; then
    echo "[ERROR]: File $1 not found"
    echo
    exit 1
fi

# By default, unless...
OUT_DIR=$(dirname "$INTRA_FILE")

if [ -n "$2" ]; then
    # if there's argument 2
    if [ -d "$2" ]; then
        # if argument 2 is a directory
        OUT_DIR="$2"
        if [[ -n "$3" && -f "$3" ]]; then
            SRC_MAP="$3"
        fi
        
    elif [ -f "$2" ]; then
        SRC_MAP="$2"
        if [[ -n "$3" && -d "$3" ]]; then
            OUT_DIR="$3"
        fi
    fi
fi

INTRA_FOI="/tmp/mine-hard-ids-inter-stats.tmp"

EXP_NAME=$(basename "$INTRA_FILE")
EXP_NAME="${EXP_NAME%.*}"

entropy_calc(){
    while read -r count id; do 
        awk -viof="$id" -vtot="$count" \
            '$2 == iof {p=$1/tot; acc=acc+p*log(p)/log(2)} END {acc = -1*acc; print iof" "acc}' \
            "$OUT_FILE-joint_sparsity.txt"
    done < "$OUT_FILE-hard_positives-independent_sparsity.txt"
}

mutual_confusion() {
    cache=()
    while read -r COUNT HP HN; do

        if [ -z "${cache[$HP]}" ]; then
            cache[$HN]="1"
        else
            continue
        fi

        awk -vhp="$HP" -vhn="$HN" -vcount="$COUNT"\
            ' (hp == $3 && hn == $2) {print count + $1" "$2" "$3}' \
            "$OUT_FILE-joint_sparsity.txt"

    done < "$OUT_FILE-joint_sparsity.txt"
}

if [ -n "$OUT_DIR" ]; then
    OUT_DIR=$(dirname "$INTRA_FILE")
fi

OUT_FILE="$OUT_DIR/$EXP_NAME"

grep -v "#" "$INTRA_FILE" > "$INTRA_FOI"

echo
echo "STATS: $EXP_NAME"
echo "-----"
echo 
echo " 1. Number of Patch Hard Positives":
wc -l "$INTRA_FOI"

echo
echo " 2. Number of Hard Positives:"
sort -nk1,1 "$INTRA_FOI" | awk '{print $1}' | uniq -c | sort -nrk1,1 \
    > "$OUT_FILE-hard_positives-independent_sparsity.txt"
wc -l "$OUT_FILE-hard_positives-independent_sparsity.txt"

echo 
echo " 3. Number of Hard Negatives:"
sort -nuk2,2 "$INTRA_FOI" | awk '{print $2}' | uniq -c | sort -nrk1,1 \
    > "$OUT_FILE-hard_negatives-attractors.txt"
wc -l "$OUT_FILE-hard_negatives-attractors.txt"

sort -nk1,1 "$INTRA_FOI" | awk '{print $1" "$2}' | uniq -c | sort -nrk1,1 > "$OUT_FILE-joint_sparsity.txt"

echo 
echo " 4. Joint Sparsity saved in $OUT_FILE-joint_sparsity.txt"

entropy_calc | sort -nrk2,2 > "$OUT_FILE-entropy.txt"
mutual_confusion > "$OUT_FILE-mutual-attraction.txt"

if [ -n "$SRC_MAP" ]; then
    echo
    echo " 6. Hard Positive images saved in $OUT_FILE-HP-map.txt"
    paste "$INTRA_FOI" <(awk 'FNR==NR {lines[$4]; next} (FNR in lines)' "$INTRA_FOI" "$SRC_MAP") > "$OUT_FILE-HP-map.txt"
fi

