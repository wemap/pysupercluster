# -*- coding: utf-8 -*-

import os

from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext as _build_ext

root_dir = os.path.abspath(os.path.dirname(__file__))
readme_file = os.path.join(root_dir, 'README.rst')
with open(readme_file, encoding='utf-8') as f:
    long_description = f.read()


class build_ext(_build_ext):
    def finalize_options(self):
        _build_ext.finalize_options(self)
        # Prevent numpy from thinking it is still in its setup process:
        __builtins__.__NUMPY_SETUP__ = False
        import numpy
        self.include_dirs.append(numpy.get_include())


setup(
    name='pysupercluster',
    version='0.7.5',
    description='A fast geospatial point clustering module.',
    long_description=long_description,
    author='Jeremy Lainé',
    author_email='jeremy@getwemap.com',
    url='https://github.com/wemap/pysupercluster',
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        "Intended Audience :: Developers",
        "License :: OSI Approved :: ISC License (ISCL)",
        "Programming Language :: C++",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3 :: Only",
        "Topic :: Scientific/Engineering :: GIS",
    ],
    cmdclass={'build_ext': build_ext},
    install_requires=['numpy'],
    python_requires='>=3',
    setup_requires=['numpy'],
    ext_modules=[
        Extension(
            'pysupercluster',
            extra_compile_args=['-std=c++1y'],
            language='c++',
            depends=[
                'src/kdbush.hpp',
                'src/supercluster.hpp',
            ],
            sources=[
                'src/module.cpp',
                'src/supercluster.cpp',
            ],
        )
    ]
)
