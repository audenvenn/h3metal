"""Builds the GPU runtime library into the Python package.

`pip install .` compiles libh3metal.dylib with CMake and Ninja (fetched by pip
as build requirements) and ships it inside the `h3metal` package together with
the Metal kernel sources, which are compiled at run time. The only thing the
machine needs is a C++ compiler: the Xcode command line tools.

Editable installs (`pip install -e .`) skip this: build with CMake yourself
and the library is found in build/cmake.
"""

import glob
import os
import shutil
import subprocess
import sys
import tempfile

from setuptools import setup
from setuptools.command.bdist_wheel import bdist_wheel
from setuptools.command.build_py import build_py

HERE = os.path.dirname(os.path.abspath(__file__))


class BuildWithRuntime(build_py):
    def run(self):
        super().run()
        if getattr(self, "editable_mode", False) or os.environ.get("H3METAL_SKIP_BUILD") == "1":
            return
        if sys.platform != "darwin":
            raise SystemExit("h3metal needs macOS on Apple Silicon")
        target = os.path.join(self.build_lib, "h3metal")
        with tempfile.TemporaryDirectory() as tmp:
            subprocess.run(["cmake", "-S", HERE, "-B", tmp, "-G", "Ninja", "-DCMAKE_BUILD_TYPE=Release"], check=True)
            subprocess.run(["cmake", "--build", tmp, "--target", "h3metal"], check=True)
            shutil.copy2(os.path.join(tmp, "libh3metal.dylib"), os.path.join(target, "libh3metal.dylib"))
        kernels = os.path.join(target, "kernels")
        os.makedirs(kernels, exist_ok=True)
        for path in glob.glob(os.path.join(HERE, "kernels", "*")):
            shutil.copy2(path, kernels)


class PlatformWheel(bdist_wheel):
    """The wheel holds a macOS arm64 library (loaded with ctypes, so any Python 3 will do)."""

    def finalize_options(self):
        super().finalize_options()
        self.root_is_pure = False

    def get_tag(self):
        _, _, plat = super().get_tag()
        return "py3", "none", plat


setup(cmdclass={"build_py": BuildWithRuntime, "bdist_wheel": PlatformWheel})
