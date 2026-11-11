# Paths and interpreters for all scripts (source this file; every script in src/ does).
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export SRC="$REPO/src"
export DATA_ROOT="${DATA_ROOT:-$REPO/data_root}"          # data_root/data/<city>/{map,traj}
export HOSER_DATA="${HOSER_DATA:-$REPO/data_raw/hoser}"   # HOSER release (Hugging Face caoji2001/HOSER-dataset)
export GTG_DIR="${GTG_DIR:-$SRC/gtg_official}"            # GTG clone (setup_third_party.sh)
export HOSER_DIR="${HOSER_DIR:-$REPO/third_party/HOSER}"  # HOSER clone (setup_third_party.sh)
export HOSER_FS="${HOSER_FS:-$REPO/third_party/hoser_fewshot}"
export PY="${PY:-conda run --no-capture-output -n gtg python}"
export PY_GTGP2="${PY_GTGP2:-conda run --no-capture-output -n gtgp2 python}"
export PY_HOSER="${PY_HOSER:-conda run --no-capture-output -n hoser python}"
export CC_CITIES="${CC_CITIES:-xian,chengdu,beijing,porto,sanfrancisco}" CC_F1="${CC_F1:-length}"
