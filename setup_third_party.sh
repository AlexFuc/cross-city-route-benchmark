#!/bin/bash
# Clone the official baselines at the pinned commits and add our drivers (no tracked upstream file is modified).
set -e
source "$(dirname "$0")/config.sh"
mkdir -p "$REPO/third_party"
git clone https://github.com/lyd1881310/GTG.git "$GTG_DIR" && git -C "$GTG_DIR" checkout 1850d67b6ecb6cdb4a74e93462ad1d03220ec1bb
ln -sfn "$DATA_ROOT/data" "$GTG_DIR/data"
cp "$REPO/patches/gtg/gtg_generate.py" "$GTG_DIR/_driver_generate.py"; cp "$REPO/patches/gtg/_bj_takeover.py" "$GTG_DIR/"
git clone https://github.com/caoji2001/HOSER.git "$HOSER_DIR" && git -C "$HOSER_DIR" checkout 92c6ca1ca6f7db4699f64ca1db554a44a73b70dc
cp "$REPO"/patches/hoser/*.py "$HOSER_DIR/"
git clone --depth 1 --branch v3.17 https://github.com/KaHIP/KaHIP.git "$HOSER_DIR/KaHIP"
( cd "$HOSER_DIR/KaHIP" && mkdir -p build && cd build && cmake .. -DCMAKE_BUILD_TYPE=Release && make -j8 kaffpa )
for d in Beijing Porto San_Francisco; do mkdir -p "$HOSER_DIR/data/$d"; cp "$HOSER_DATA/$d"/*.csv "$HOSER_DATA/$d"/roadmap.* "$HOSER_DIR/data/$d/"; done
( cd "$HOSER_DIR/data/preprocess" && $PY_HOSER partition_road_network.py && $PY_HOSER get_zone_trans_mat.py )
echo "third-party setup done"
