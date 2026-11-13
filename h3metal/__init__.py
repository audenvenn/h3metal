"""MiniMax-H3-Image on Apple Silicon: a Metal engine for the GPU neural accelerators.

    from h3metal import Pipeline
    pipe = Pipeline("~/.h3metal/model")
    image, timing = pipe.generate("A red fox in fresh snow", 2048, 2048, seed=0)
"""

__version__ = "0.5.0"


def __getattr__(name):
    if name == "Pipeline":  # imported on first use: the package itself loads nothing heavy
        from .pipeline import Pipeline
        return Pipeline
    raise AttributeError(name)
