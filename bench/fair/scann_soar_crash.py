"""Small reproduction of ScaNN 1.4.2 crashing when a SOAR index is updated.

    python bench/fair/scann_soar_crash.py N VARIANT

VARIANT is a dash-separated list: soar or nosoar first, then int (integer docids) or str
(string docids), and optionally int8 (INT8 reorder instead of f32), bs1 (upsert with
batch_size 1) or prefix (delete the first ids instead of random ones). For example soar-int,
soar-int-int8, nosoar-int, soar-str, soar-int-bs1.

Each run builds on N rows, then does 10 rounds of deleting and upserting N/40 vectors. What I
see: with SOAR and f32 reorder, delete() segfaults; with SOAR and INT8 reorder, upsert() corrupts
the heap at N=400k; without SOAR it finishes.

The process marks itself non-dumpable first so a crash doesn't leave a core dump, and
faulthandler prints the Python stack.
"""
import ctypes
import faulthandler
import os
import sys
import time

ctypes.CDLL(None).prctl(4, 0, 0, 0, 0)       # PR_SET_DUMPABLE = 4 -> 0: no core dump
faulthandler.enable()
import numpy as np  # noqa: E402
import scann  # noqa: E402

n, variant = int(sys.argv[1]), sys.argv[2]
X = np.ascontiguousarray(np.load(os.path.join(os.environ.get("FAIR_DIR", os.path.expanduser("~/maivs-fair")), "data",
                                      "base_500k.npy"), mmap_mode="r")
                         [:n + n // 4], np.float32)
soar = variant.startswith("soar")
conv = str if "-str" in variant else int
ubs = 1 if variant.endswith("bs1") else 256
nd = n // 4
rng = np.random.default_rng(1)
dele = np.arange(nd) if "prefix" in variant else rng.choice(n, nd, replace=False)
from scann.scann_ops.py.scann_builder import ReorderType  # noqa: E402
rq = ReorderType.INT8 if "int8" in variant else ReorderType.FLOAT32

b = scann.scann_ops_pybind.builder(X[:n], 10, "dot_product").set_n_training_threads(16)
kw = dict(num_leaves=max(20, n // 250), num_leaves_to_search=32, training_sample_size=min(n, 250_000))
if soar:
    kw.update(soar_lambda=1.5, overretrieve_factor=2.0)
t0 = time.time()
s = b.tree(**kw).score_ah(2, anisotropic_quantization_threshold=0.2).reorder(100, quantize=rq).build(
    docids=[conv(i) for i in range(n)])
print(f"[{variant} n={n}] built in {time.time() - t0:.1f}s size {s.size()}", flush=True)
step = nd // 10
for r in range(10):
    d = dele[r * step:(r + 1) * step]
    new = np.arange(n + r * step, n + (r + 1) * step)
    print(f"  round {r}: delete {len(d)}", flush=True)
    s.delete([conv(i) for i in d])
    print(f"  round {r}: upsert {len(new)}  (batch_size {ubs})", flush=True)
    s.upsert([conv(i) for i in new], np.ascontiguousarray(X[new]), batch_size=ubs)
    print(f"  round {r}: ok, size {s.size()}", flush=True)
I, _ = s.search_batched_parallel(X[:5], final_num_neighbors=10, leaves_to_search=32,
                                 pre_reorder_num_neighbors=100, batch_size=128)
print(f"[{variant} n={n}] churn completed, search ok", flush=True)
os._exit(0)
