#include <algorithm>
#include <chrono>

#include "supercluster.hpp"


Cluster::Cluster(const Point &_point, size_t _numPoints, size_t _id)
    : point(_point), numPoints(_numPoints), id(_id), zoom(999)
{
}


ClusterTree::ClusterTree(const std::vector<Cluster*> &_clusters)
    : clusters(_clusters)
{
    std::vector<Point> points;
    points.reserve(clusters.size());
    for (size_t i = 0; i < clusters.size(); ++i)
        points.push_back(clusters[i]->point);
    kdbush = new kdbush::KDBush<Point>(points);
}


ClusterTree::~ClusterTree()
{
    delete kdbush;
}


SuperCluster::SuperCluster(const std::vector<Point> &points)
{
    auto start = std::chrono::steady_clock::now();

    trees.resize(maxZoom + 2);

    // prepare initial clusters
    std::vector<Cluster*> clusters;
    clusters.reserve(points.size());
    for (size_t i = 0; i < points.size(); ++i)
        clusters.push_back(new Cluster(points[i], 1, i));
    all_clusters = clusters;

    for (int z = maxZoom; z >= minZoom; --z) {
        auto zoom_start = std::chrono::steady_clock::now();

        trees[z + 1] = new ClusterTree(clusters);

        clusters = cluster(clusters, z);
        
        auto elapsed = std::chrono::steady_clock::now() - zoom_start;
        printf("z%d: %ld clusters in %ldms\n", z, clusters.size(), std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count());
    }

    auto elapsed = std::chrono::steady_clock::now() - start;
    printf("total time: %ldms\n", std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count());
}


SuperCluster::~SuperCluster()
{
    for (size_t i = 0; i < trees.size(); ++i)
        delete trees[i];
    for (size_t i = 0; i < all_clusters.size(); ++i)
        delete all_clusters[i];
}


std::vector<Cluster*> SuperCluster::cluster(const std::vector<Cluster*> &points, int zoom)
{
    std::vector<Cluster*> clusters;
    double radius = this->radius / (this->extent * (1 << zoom));
    ClusterTree *tree = trees[zoom + 1];

    for (size_t i = 0; i < points.size(); ++i) {
        Cluster *p = points[i];
        if (p->zoom <= zoom)
            continue;
        p->zoom = zoom;

        bool foundNeighbors = false;
        size_t numPoints = p->numPoints;
        double wx = p->point.first * numPoints;
        double wy = p->point.second * numPoints;

        tree->kdbush->within(p->point.first, p->point.second, radius, [&foundNeighbors, &numPoints, tree, &wx, &wy, zoom](const auto id) {
            Cluster *b = tree->clusters[id];
            if (zoom < b->zoom) {
                foundNeighbors = true;
                b->zoom = zoom;
                wx += b->point.first * b->numPoints;
                wy += b->point.second * b->numPoints;
                numPoints += b->numPoints;
            }
        });

        if (foundNeighbors) {
            Cluster *cluster = new Cluster(Point(wx / numPoints, wy / numPoints), numPoints, -1);
            clusters.push_back(cluster);
            all_clusters.push_back(cluster);
        } else {
            clusters.push_back(p);
        }
    }

    return clusters;
}


std::vector<Cluster*> SuperCluster::getClusters(const Point &min_p, const Point &max_p, int zoom) const
{
    const int z = std::max(minZoom, std::min(zoom, maxZoom));
    std::vector<Cluster*> clusters;

    ClusterTree *tree = trees[z + 1];
    tree->kdbush->range(min_p.first, min_p.second, max_p.first, max_p.second, [&clusters, &tree](const auto id) {
        clusters.push_back(tree->clusters[id]);
    });

    return clusters;
}
