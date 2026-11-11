"""Inspect HOSER-format city data: columns, route connectivity via roadmap.rel, timestamps, sizes."""
import sys

import pandas as pd

city = sys.argv[1]
split = sys.argv[2] if len(sys.argv) > 2 else "train"
base = f"hoser/{city}"
df = pd.read_csv(f"{base}/{split}.csv", nrows=200000)
print("columns:", list(df.columns))
print(df.head(2).to_string()[:1500])
rel = pd.read_csv(f"{base}/roadmap.rel")
succ = set(zip(rel.origin_id, rel.destination_id))
col = [c for c in df.columns if "rid" in c.lower() or "road" in c.lower() or "list" in c.lower()]
print("candidate route columns:", col)
rc = col[0]
tot = adj = 0
lens = []
for s in df[rc].head(20000):
    r = [int(x) for x in str(s).strip("[]").replace(" ", "").split(",") if x != ""]
    lens.append(len(r))
    for a, b in zip(r, r[1:]):
        tot += 1
        adj += (a, b) in succ
print(f"{city}/{split}: consecutive segments connected {adj / max(tot, 1):.4f} ({adj}/{tot}); "
      f"median segs {pd.Series(lens).median()}, p90 {pd.Series(lens).quantile(.9)}")
