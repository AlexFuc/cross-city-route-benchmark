"""Convert HOSER-format city data (Porto, San Francisco) into the GTG data layout used by our pipeline.

Output: root/data/<city>/{map/road.csv, map/edge.csv, map/rid_to_gps.json, traj/{train,test}.csv, traj/users.csv}
  road.csv   link_id,length,link_type_name
  edge.csv   src,trg                              (segment adjacency from roadmap.rel)
  rid_to_gps {rid: [lon, lat]} segment midpoint
  traj csv   traj_id;start_time;rid_list;dur_list (';'-separated, like the GTG data)
Time convention: existing code derives local hour as ((start_time + 8h) mod 1 day). To keep that code
unchanged, start_time is stored as (local wall-clock time interpreted as UTC epoch) - 8h, where local wall
clock comes from the IANA zone (DST-aware). Documented in root/data/README_EXT.md.
"""
import os
import ast
import json
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

HOSER_CODE = {0: "motorway", 1: "trunk", 2: "primary", 3: "secondary", 4: "tertiary", 5: "unclassified",
              6: "residential", 7: "motorway_link", 8: "trunk_link", 9: "primary_link", 10: "secondary_link",
              11: "tertiary_link", 12: "living_street"}
CITIES = {"porto": ("Porto", "Europe/Lisbon"), "sanfrancisco": ("San_Francisco", None), "bjhoser": ("Beijing", "Asia/Shanghai")}  # None: timestamps already local wall clock (see README)
SRC = Path(os.environ["HOSER_DATA"])
ROOT = Path(os.environ["DATA_ROOT"]) / "data"


def highway_name(v):
    if isinstance(v, (int, np.integer)) or (isinstance(v, str) and v.isdigit()):
        return HOSER_CODE.get(int(v), "other")
    s = str(v)
    if s.startswith("["):
        try:
            return str(ast.literal_eval(s)[0])
        except (ValueError, SyntaxError):
            return "other"
    return s


def convert_map(src, dst):
    geo = pd.read_csv(src / "roadmap.geo")
    assert (geo.geo_id.to_numpy() == np.arange(len(geo))).all(), "geo_id must be 0..N-1"
    road = pd.DataFrame({"link_id": geo.geo_id, "length": geo.length.astype(float),
                         "link_type_name": [highway_name(v) for v in geo.highway]})
    (dst / "map").mkdir(parents=True, exist_ok=True)
    road.to_csv(dst / "map/road.csv", index=False)
    rel = pd.read_csv(src / "roadmap.rel")
    rel[["origin_id", "destination_id"]].rename(columns={"origin_id": "src", "destination_id": "trg"}) \
        .drop_duplicates().to_csv(dst / "map/edge.csv", index=False)
    gps = {}
    for gid, c in zip(geo.geo_id, geo.coordinates):
        pts = json.loads(c)
        mid = pts[len(pts) // 2]
        gps[str(int(gid))] = [float(mid[0]), float(mid[1])]
    (dst / "map/rid_to_gps.json").write_text(json.dumps(gps))
    print(f"  map: {len(road)} segments, {len(rel)} relations; types {road.link_type_name.value_counts().head(6).to_dict()}")


def convert_traj(src, dst, split, tz, id_offset):
    cols = pd.read_csv(src / f"{split}.csv", nrows=0).columns
    uid = "user_id" if "user_id" in cols else "entity_id"
    df = pd.read_csv(src / f"{split}.csv", usecols=[uid, "rid_list", "time_list"]).rename(columns={uid: "user_id"})
    t0 = pd.to_datetime(df.time_list.str.split(",", n=1).str[0], utc=True)
    local_wall = t0.dt.tz_localize(None) if tz is None else t0.dt.tz_convert(ZoneInfo(tz)).dt.tz_localize(None)
    start = ((local_wall - pd.Timestamp("1970-01-01")) // pd.Timedelta(seconds=1)).astype(np.int64) - 8 * 3600
    parts = df.time_list.str.split(",")
    lens = parts.str.len().to_numpy()
    ts = (pd.to_datetime(pd.Series(np.concatenate(parts.to_numpy())), utc=True).astype("int64") // 10**9).to_numpy()
    off = np.r_[0, np.cumsum(lens)]
    durs = []
    for i in range(len(df)):
        t = ts[off[i]:off[i + 1]]
        d = np.diff(t)
        d = np.r_[d, d[-1] if len(d) else 0]
        durs.append(",".join(map(str, d.astype(np.int64))))
    out = pd.DataFrame({"traj_id": np.arange(len(df)) + id_offset, "start_time": start,
                        "rid_list": df.rid_list, "dur_list": durs})
    (dst / "traj").mkdir(parents=True, exist_ok=True)
    out.to_csv(dst / f"traj/{split}.csv", sep=";", index=False)
    return pd.DataFrame({"traj_id": out.traj_id, "user_id": df.user_id, "split": split})


def main():
    for city in sys.argv[1:]:
        name, tz = CITIES[city]
        src, dst = SRC / name, ROOT / city
        print(f"[{city}] converting")
        convert_map(src, dst)
        users = []
        if (src / "train.csv").exists():
            users.append(convert_traj(src, dst, "train", tz, 0))
        users.append(convert_traj(src, dst, "test", tz, 10_000_000))
        pd.concat(users).to_csv(dst / "traj/users.csv", index=False)
        te = pd.read_csv(dst / "traj/test.csv", sep=";", nrows=5)
        print(te.head(2).to_string()[:400])
        print(f"[{city}] done: train {sum(u.split.eq('train').sum() for u in users)}, test {sum(u.split.eq('test').sum() for u in users)}")


if __name__ == "__main__":
    main()
