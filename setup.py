import numpy
from distutils.core import setup, Extension

extensions = [
    Extension('supercluster',
        extra_compile_args=['-std=c++1y'],
        sources=['module.cpp', 'supercluster.cpp'],
    )
]

setup(name='supercluster',
      version='0.1',
      description='A crazy fast geospatial point clustering module.',
      include_dirs=[numpy.get_include()],
      ext_modules=extensions)
