supercluster
============

A very fast Python 3 module for geospatial point clustering.

This is a port of https://github.com/mapbox/supercluster.

.. code-block:: pycon

    >>> import numpy
    >>> import supercluster
    >>> points = numpy.array([
    ...     (2.3522, 48.8566),
    ...     (-0.1278, 51.5074),
    ... ])
    >>> index = supercluster.SuperCluster(
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
