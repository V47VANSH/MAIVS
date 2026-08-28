"""Cut the benchmark slices out of hotpot_5M.npy and compute exact ground truth once.

    python bench/fair/prep_data.py

Same split as bench/bench_vs_faiss.py:
  base(n) = rows [0, n) of hotpot_5M.npy
  IND     = the last 1,000 rows of hotpot_5M.npy (never indexed)
  OOD     = the 1,000 real HotpotQA questions (bench/data/hotpot_questions_minilm.npy)
The 8 GB corpus is read with seeks, never loaded or memory-mapped whole. Set FAIR_CORPUS if it
isn't in the repository root. Output goes to $FAIR_DIR/data (default ~/maivs-fair/data).
"""
import os, sys, time
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CORPUS = os.environ.get("FAIR_CORPUS", os.path.join(REPO, "hotpot_5M.npy"))
QUESTIONS = os.path.join(REPO, "bench", "data", "hotpot_questions_minilm.npy")
OUT = os.path.join(os.environ.get("FAIR_DIR", os.path.expanduser("~/maivs-fair")), "data")
NMAX = 500_000
K = 10
os.makedirs(OUT, exist_ok=True)


def read_rows(path, start, stop):
    with open(path, "rb") as f:
        ver = np.lib.format.read_magic(f)
        rd = np.lib.format.read_array_header_1_0 if ver == (1, 0) else np.lib.format.read_array_header_2_0
        shape, fortran, dtype = rd(f)
        assert not fortran and dtype == np.float32 and len(shape) == 2, (shape, dtype)
        n, d = shape
        if start < 0:
            start += n
            stop = n if stop in (None, 0) else stop + n
        off = f.tell()
        f.seek(off + start * d * 4)
        buf = np.empty((stop - start, d), np.float32)
        f.readinto(memoryview(buf).cast("B"))
        return buf, shape


t0 = time.time()
base, shape = read_rows(CORPUS, 0, NMAX)
ind, _ = read_rows(CORPUS, -1000, None)
ood = np.ascontiguousarray(np.load(QUESTIONS)[:1000], np.float32)
print(f"source {shape}, read {time.time() - t0:.1f}s", flush=True)
for name, a in (("base", base), ("IND", ind), ("OOD", ood)):
    nr = np.linalg.norm(a, axis=1)
    print(f"  {name}: {a.shape}  norm min {nr.min():.4f} max {nr.max():.4f}", flush=True)
np.save(os.path.join(OUT, "base_500k.npy"), base)
np.save(os.path.join(OUT, "IND.npy"), ind)
np.save(os.path.join(OUT, "OOD.npy"), ood)


def exact_gt(b, Q, chunk=131072):
    best_s = np.full((len(Q), K), -9.0, np.float32)
    best_i = np.zeros((len(Q), K), np.int64)
    for s in range(0, len(b), chunk):
        S = Q @ b[s:s + chunk].T
        idx = np.argpartition(-S, K, axis=1)[:, :K]
        sc = np.take_along_axis(S, idx, 1)
        cs = np.concatenate([best_s, sc], 1)
        ci = np.concatenate([best_i, idx + s], 1)
        o = np.argsort(-cs, axis=1)[:, :K]
        best_s = np.take_along_axis(cs, o, 1)
        best_i = np.take_along_axis(ci, o, 1)
    return best_i


for n in (250_000, 500_000):
    t0 = time.time()
    np.savez(os.path.join(OUT, f"gt_{n}.npz"), IND=exact_gt(base[:n], ind), OOD=exact_gt(base[:n], ood))
    print(f"gt n={n:,}: {time.time() - t0:.1f}s", flush=True)
print("done")
