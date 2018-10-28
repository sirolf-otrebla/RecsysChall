from distutils.core import setup

from Cython.Build import cythonize
from setuptools import setup
from setuptools import Extension
from Cython.Distutils import build_ext

ext_modules = cythonize(Extension(
        "MF_RMSE",
        sources=["MF_RMSE.pyx"],
        language="c",
))
