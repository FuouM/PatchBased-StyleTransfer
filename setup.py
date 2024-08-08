from setuptools import setup, Extension
import sys
import numpy

module = Extension(
    "gaussian_mixture",
    sources=["src/gaussian_mixture.cpp"],
    include_dirs=[numpy.get_include()], 
    extra_compile_args=["/std:c++17"] if sys.platform == "win32" else ["-std=c++17"],
)

setup(
    name="gaussian_mixture",
    version="1.0",
    description="Test module",
    ext_modules=[module],
)
