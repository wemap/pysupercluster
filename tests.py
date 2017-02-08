import unittest

import numpy

import pysupercluster


class SuperClusterTest(unittest.TestCase):
    def test_clustering(self):
        points = numpy.array([
            (2.3522, 48.8566),   # paris
            (-0.1278, 51.5074),  # london
            (-0.0077, 51.4826),  # greenwhich
        ])

        index = pysupercluster.SuperCluster(
            points,
            min_zoom=0,
            max_zoom=16,
            radius=40,
            extent=512)

        clusters = index.getClusters(
            top_left=(-180, 90),
            bottom_right=(180, -90),
            zoom=4)

        self.assertEqual(len(clusters), 2)

        self.assertEqual(clusters[0]['count'], 1)
        self.assertEqual(clusters[0]['id'], 0)

        self.assertEqual(clusters[1]['count'], 2)
        self.assertEqual(clusters[1]['id'], 3)


if __name__ == '__main__':
    unittest.main()
