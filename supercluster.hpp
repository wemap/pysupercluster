#include "kdbush.hpp"

using Point = std::pair<double, double>;

class Cluster {
public:
    Cluster(const Point &_point, size_t _numPoints, size_t _id);
    Point point;
    size_t numPoints;
    size_t id;
    int zoom;
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
    SuperCluster(const std::vector<Point> &points);
    ~SuperCluster();

    std::vector<Cluster*> cluster(const std::vector<Cluster*> &points, int zoom);

private:
    const int minZoom = 0;
    const int maxZoom = 16;
    const double radius = 40;
    const double extent = 512;

    std::vector<Cluster*> all_clusters;
    std::vector<ClusterTree*> trees;
};
