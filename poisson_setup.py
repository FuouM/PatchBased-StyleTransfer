from setuptools import setup, Extension
import sys

module = Extension(
    "poisson_disk_module",
    sources=["src/poisson_wrapper.cpp"],
    extra_compile_args=["/std:c++17"] if sys.platform == "win32" else ["-std=c++17"],
)

setup(
    name="poisson_disk_module",
    version="1.0",
    description="Poisson disk sampling module",
    ext_modules=[module],
)
