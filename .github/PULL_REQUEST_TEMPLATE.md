## What this changes

<!-- One or two sentences. Link the issue it addresses, if there is one. -->

## How it was checked

<!-- Tick what you ran and paste the relevant output. Speed claims need the machine, the commands, and a back-to-back baseline on an idle GPU. -->

- [ ] `python tests/test_gemm.py`, `tests/test_attention.py`, `tests/test_ops.py` (after touching `kernels/` or `src/`)
- [ ] The model tests against the official modules (`tests/test_dit_mini.py`, `tests/test_te_mini.py`, `tests/test_vae.py`)
- [ ] `bench/quality.py` against a reference run (after changing numerics)
- [ ] Benchmarks, if the change is about speed

## Checklist

- [ ] Code, comments and documentation are in English
- [ ] `CHANGELOG.md` has an entry under "Unreleased" if users will notice the change
