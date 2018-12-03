from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy as np                           # <---- New line

ext_modules = [Extension("SLIM_BPR_Cython_Epoch", ["SLIM_BPR_Cython_Epoch.pyx"])]

setup(
  name = 'SLIM_BPR_Cython_Epoch',
  cmdclass = {'build_ext': build_ext},
  include_dirs = [np.get_include()],         # <---- New line
  ext_modules = ext_modules
)