"""Add a turn-angle column to converted HOSER cities, same definition as GTG's prepare.py:
angle(src->trg) = heading(trg) - heading(src), heading = atan2 of the whole segment's start->end vector
(planar approximation x = lon * cos(lat0), y = lat)."""
import os
import json
import sys

import numpy as np
import pandas as pd

NAMES = {"porto": "Porto", "sanfrancisco": "San_Francisco"}
for city in sys.argv[1:]:
    geo = pd.read_csv(os.path.join(os.environ["HOSER_DATA"], NAMES[city], "roadmap.geo"), usecols=["geo_id", "coordinates"])
    pts = [json.loads(c) for c in geo.coordinates]
    lat0 = np.mean([p[0][1] for p in pts])
    k = np.cos(np.deg2rad(lat0))
    head = np.array([np.arctan2(p[-1][1] - p[0][1], (p[-1][0] - p[0][0]) * k) for p in pts])
    path = os.path.join(os.environ["DATA_ROOT"], "data", city, "map", "edge.csv")
    e = pd.read_csv(path, usecols=["src", "trg"])
    e["angle"] = head[e.trg.to_numpy()] - head[e.src.to_numpy()]
    e.to_csv(path, index=False)
    w = (e.angle + np.pi) % (2 * np.pi) - np.pi
    print(city, len(e), "uturn(>150deg) share", round(float(np.mean(np.abs(w) > np.deg2rad(150))), 3))
