"""Road-graph helpers used by every experiment (shared by all experiments).

Segment graph: vertices = road segments, edges = permitted transitions (data/<city>/map/edge.csv).
A route cost is the sum of per-segment costs; as the origin and destination segments are fixed per query, an edge
weight of (cost[src] + cost[dst]) / 2 yields the same shortest paths as summing segment costs.
LEN-SP uses RAW segment lengths (no clipping or winsorization).
"""
from functools import lru_cache

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

import diag


@lru_cache(maxsize=None)
def city_graph(city):
    e = pd.read_csv(diag.DATA_ROOT / f"data/{city}/map/edge.csv", usecols=["src", "trg"], dtype={"src": np.int32, "trg": np.int32})
    r = pd.read_csv(diag.DATA_ROOT / f"data/{city}/map/road.csv", usecols=["length"], dtype={"length": np.float64})
    return int(len(r)), e["src"].to_numpy(np.int32), e["trg"].to_numpy(np.int32), r["length"].to_numpy(np.float64)


def raw_length(city):
    return city_graph(city)[3].copy()


def csr_from_node_cost(city, node_cost):
    n, src, dst, _ = city_graph(city)
    node_cost = np.asarray(node_cost, np.float64)
    assert node_cost.shape[0] == n, (node_cost.shape, n)
    return csr_matrix(((node_cost[src] + node_cost[dst]) / 2.0, (src, dst)), shape=(n, n))


def hausdorff_metric(truth, pred):
    """Symmetric Hausdorff distance (km) between two (lat, lon) point sequences, haversine, R = 6378 km
    (same formula as the HOSER/GTG evaluation code)."""
    if len(truth) == 0 or len(pred) == 0:
        return 0.0
    R, rad = 6378.0, np.pi / 180.0
    lat1, lon1 = truth[:, 0:1] * rad, truth[:, 1:2] * rad
    lat2, lon2 = pred[:, 0:1].T * rad, pred[:, 1:2].T * rad
    a = np.sin((lat2 - lat1) / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2.0) ** 2
    d = R * 2.0 * np.arcsin(np.sqrt(a))
    return float(max(np.max(np.min(d, axis=1)), np.max(np.min(d, axis=0))))
