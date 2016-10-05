pysupercluster
==============

A fast Python 3 module for geospatial point clustering.

This is a port of https://github.com/mapbox/supercluster to C++, conveniently
wrapped in a Python module. Initial benchmarks show it to be an order of
magnitude (10x) faster than the original JavaScript implementation.

Installing pysupercluster
-------------------------

The easiest way to install pysupercluster is to use pip:

    pip install pysupercluster

Using pysupercluster
--------------------

.. code-block:: pycon

    >>> import numpy
    >>> import pysupercluster
    >>> points = numpy.array([
    ...     (2.3522, 48.8566),   # paris
    ...     (-0.1278, 51.5074),  # london
    ...     (-0.0077, 51.4826),  # greenwhich
    ... ])
    >>> index = pysupercluster.SuperCluster(
    ...     points,
    ...     min_zoom=0,
    ...     max_zoom=16,
    ...     radius=40,
    ...     extent=512)
    >>> clusters = index.getClusters(
    ...     top_left=(-180, 90),
    ...     bottom_right=(180, -90),
    ...     zoom=4)
    [
        {'id': 0, 'count': 1, 'latitude': 48.8566, 'longitude': 2.3522},
        {'id': None, 'count': 2, 'latitude': 51.49500168658321, 'longitude': -0.06774999999998421}
    ]
