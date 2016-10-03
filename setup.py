import numpy
from distutils.core import setup, Extension

extensions = [
    Extension('supercluster',
        extra_compile_args=['-std=c++1y'],
        sources=['src/module.cpp', 'src/supercluster.cpp'],
    )
]

setup(name='supercluster',
      version='0.2',
      description='A fast geospatial point clustering module.',
      include_dirs=[numpy.get_include()],
      ext_modules=extensions)
