"""Wall-clock cost of every official GTG stage per direction, from the [INFO](HH:MM:SS) stamps of the official
logs (day rollovers handled by monotonicity) plus the takeover logs for the two Beijing-source directions."""
import json
import re
import sys
from pathlib import Path

D = Path(__file__).resolve().parent
STAMP = re.compile(rb"\[INFO\]\((\d\d):(\d\d):(\d\d)\):([^\r\n]*)")


def events(path):
    out, day, last = [], 0, -1
    for m in STAMP.finditer(path.read_bytes()):
        s = int(m[1]) * 3600 + int(m[2]) * 60 + int(m[3])
        if s < last:
            day += 1
        last = s
        out.append((day * 86400 + s, m[4].decode(errors="replace")))
    return out


def first(ev, pat):
    return next((t for t, m in ev if pat in m), None)


def lastt(ev, pat):
    return next((t for t, m in reversed(ev) if pat in m), None)


res = {}
for src, dst, ab in [("chengdu", "xian", "CD->XA"), ("xian", "chengdu", "XA->CD"), ("xian", "beijing", "XA->BJ"),
                     ("chengdu", "beijing", "CD->BJ"), ("beijing", "xian", "BJ->XA"), ("beijing", "chengdu", "BJ->CD")]:
    name = "train_CD2XA.log" if ab == "CD->XA" else f"train_{src}2{dst}.log"
    ev = events(D / "gtg_official" / name)
    t0 = first(ev, "Disentangle train, epoch 0 ")
    r = {}
    if src != "beijing":
        t599 = lastt(ev, "Disentangle train, epoch 599")
        tend = ev[-1][0]
        r["disentangle_h"] = round((t599 - t0) / 3600 * 600 / 599, 2)
        r["preference_h"] = round((tend - t599) / 3600, 2)
        r["pref_end_marker"] = ev[-1][1][:80]
    else:
        t595 = lastt(ev, "Disentangle train, epoch 595")
        tk = events(D / f"takeover_{src}2{dst}.log")
        k596, k599 = first(tk, "epoch 596"), lastt(tk, "epoch 599")
        kdone = lastt(tk, "TAKEOVER DONE")
        per_epoch_k = (k599 - k596) / 3
        r["disentangle_h"] = round(((t595 - t0) * 596 / 595 + 4 * per_epoch_k) / 3600, 2)
        r["preference_h"] = round((kdone - k599) / 3600, 2)
        r["note"] = "epochs 596-599 and preference training via takeover (vectorized build_graph)"
    g = (D / f"gen_{src}2{dst}.log").read_bytes().replace(b"\r", b"\n")
    el = re.findall(rb"\[(\d+):(\d+)(?::(\d+))?<", g)
    if el:   # last tqdm elapsed of the official generation (100k ODs): [mm:ss< or [h:mm:ss<
        a, b, c = el[-1]
        r["generate_s"] = int(a) * 3600 + int(b) * 60 + int(c) if c else int(a) * 60 + int(b)
    res[ab] = r
print(json.dumps(res, indent=1))
(D / "results/GTG_costs.json").write_text(json.dumps(res, indent=2))
