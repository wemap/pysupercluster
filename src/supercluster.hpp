/*
    Copyright (c) 2016, Wemap SAS

    Permission to use, copy, modify, and/or distribute this software for any
    purpose with or without fee is hereby granted, provided that the above
    copyright notice and this permission notice appear in all copies.

    THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES
    WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
    MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR
    ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
    WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
    ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF
    OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.
*/

#include "kdbush.hpp"

using Point = std::pair<double, double>;

class Cluster {
public:
    Cluster(const Point &_point, size_t _numPoints, size_t _id, int _expansionZoom);
    Point point;
    size_t numPoints;
    size_t id;
    int zoom;
    int expansionZoom;
};


class ClusterTree {
public:
    ClusterTree(const std::vector<Cluster*> &clusters);
    ~ClusterTree();

    kdbush::KDBush<Point> *kdbush;
    std::vector<Cluster*> clusters;
};


class SuperCluster {
public:
    SuperCluster(const std::vector<Point> &points, int minZoom, int maxZoom, double radius, double extent);
    ~SuperCluster();

    std::vector<Cluster*> getClusters(const Point &min_p, const Point &max_p, int zoom) const;

private:
    std::vector<Cluster*> cluster(const std::vector<Cluster*> &points, int zoom);

    const int minZoom;
    const int maxZoom;
    const double radius;
    const double extent;

    std::vector<Cluster*> all_clusters;
    std::vector<ClusterTree*> trees;
};
