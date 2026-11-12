import os, sys, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from h3metal.gpu import Gpu
gpu = Gpu(os.path.join(os.path.dirname(__file__), "metal"))
for TM, TN, NSG in [(64, 128, 4), (128, 128, 8), (16, 32, 1)]:
    k = gpu.kernel("layout.metal", "layout", TM=TM, TN=TN, NSG=NSG)
    a = gpu.empty((TM, 128), "i8"); w = gpu.empty((TN, 64), "u8"); out = gpu.empty((NSG * 32 * 256 * 2 + 1,), "i32")
    gpu.launch(k, (1, 1, 1), (32 * NSG, 1, 1), [a, w, out]); gpu.wait()
    o = out.numpy(); cap = o[-1]
    lay = o[:-1].reshape(NSG * 32, 256, 2)[:, :cap]
    print(f"TM={TM} TN={TN} NSG={NSG}: capacity {cap}")
    for t in [0, 1, 2, 3, 4, 8, 32, 33, 64]:
        if t < NSG * 32:
            v = lay[t]; v = v[v[:, 0] >= 0]
            print(f"  thread {t:3d}: cols {sorted(set(v[:,0].tolist()))[:20]} rows {sorted(set(v[:,1].tolist()))[:20]} n={len(v)}")
print("element order, TM=64 TN=128 NSG=4, thread 0 and 1:")
k = gpu.kernel("layout.metal", "layout", TM=64, TN=128, NSG=4)
a = gpu.empty((64, 128), "i8"); w = gpu.empty((128, 64), "u8"); out = gpu.empty((4 * 32 * 256 * 2 + 1,), "i32")
gpu.launch(k, (1, 1, 1), (128, 1, 1), [a, w, out]); gpu.wait()
lay = out.numpy()[:-1].reshape(128, 256, 2)[:, :64]
for t in (0, 1, 2, 32, 64):
    print(t, [tuple(x) for x in lay[t].tolist()])
