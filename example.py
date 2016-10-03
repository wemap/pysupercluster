import numpy
import supercluster

points = numpy.array([
    (2.3522, 48.8566),   # paris
    (-0.1278, 51.5074),  # london
    (-0.0077, 51.4826),  # greenwhich
])

index = supercluster.SuperCluster(
    points,
    min_zoom=0,
    max_zoom=16,
    radius=40,
    extent=512)

clusters = index.getClusters(
    top_left=(-180, 90),
    bottom_right=(180, -90),
    zoom=4)

print(clusters)
