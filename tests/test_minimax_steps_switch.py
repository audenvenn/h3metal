"""One MiniMax-H3-Image pipeline that switches between its transformer files must give each the same image as a
fresh pipeline: the cached text rows come from a file's own token refiner (pDMD's adapter changes it), so they must
not cross over. Needs a converted model with both files (h3metal convert ... and convert ... --pdmd).

    python tests/test_minimax_steps_switch.py [MODEL_DIR]
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from h3metal.pipeline import Pipeline  # noqa: E402

model = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser("~/.h3metal/model")
prompt = "A red fox sitting in fresh snow at sunrise, photorealistic, soft golden light."
size = dict(width=1024, height=1024, seed=42, min_area=1024 * 1024)
pipe = Pipeline(model, log=lambda *a: None)
pipe.generate(prompt, steps=6, **size)
switched, _ = pipe.generate(prompt, steps=2, **size)
fresh, _ = Pipeline(model, log=lambda *a: None).generate(prompt, steps=2, **size)
diff = int(np.abs(switched.astype(int) - fresh).max())
print(f"2 steps after 6 in one pipeline against a fresh pipeline: largest pixel difference {diff}")
assert diff == 0
print("ok")
