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

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define _USE_MATH_DEFINES

#include <Python.h>
#include <numpy/arrayobject.h>

#include <cmath>
#include "supercluster.hpp"


static double lngX(double lng) {
    return lng / 360.0 + 0.5;
}


static double latY(double lat) {
    if (lat <= -90)
        return 1.0;
    else if (lat >= 90)
        return 0.0;
    else {
        double sin = std::sin(lat * M_PI / 180);
        return 0.5 - 0.25 * std::log((1 + sin) / (1 - sin)) / M_PI;
    }
}


static double xLng(double x) {
    return (x - 0.5) * 360;
}


static double yLat(double y) {
    double y2 = (180 - y * 360) * M_PI / 180;
    return 360 * std::atan(std::exp(y2)) / M_PI - 90;
}


typedef struct {
    PyObject_HEAD
    SuperCluster *sc;
} SuperClusterObject;


static int
SuperCluster_init(SuperClusterObject *self, PyObject *args, PyObject *kwargs)
{
    const char *kwlist[] = {"points", "min_zoom", "max_zoom", "radius", "extent", NULL};

    PyArrayObject *points;
    int min_zoom = 0;
    int max_zoom = 16;
    double radius = 40;
    double extent = 512;

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O!|iidd", const_cast<char **>(kwlist), &PyArray_Type, &points,
                                     &min_zoom, &max_zoom, &radius, &extent))
        return -1;

    if (PyArray_DESCR(points)->type_num != NPY_DOUBLE || PyArray_NDIM(points) != 2 || PyArray_DIMS(points)[1] != 2 || PyArray_DIMS(points)[0] == 0){
        PyErr_SetString(PyExc_ValueError, "Array must be of type double and 2 dimensional and must have a length >= 1.");
        return -1;
    }

    npy_intp count = PyArray_DIMS(points)[0];
    std::vector<Point> items(count);
    for (npy_intp i = 0; i < count; ++i) {
        items[i] = std::make_pair(
            lngX(*(double*)PyArray_GETPTR2(points, i, 0)),
            latY(*(double*)PyArray_GETPTR2(points, i, 1)));
    }
    self->sc = new SuperCluster(items, min_zoom, max_zoom, radius, extent);

    return 0;
}


static void
SuperCluster_dealloc(SuperClusterObject *self)
{
    delete self->sc;
    Py_TYPE(self)->tp_free((PyObject *) self);
}


static PyObject *
SuperCluster_getClusters(SuperClusterObject *self, PyObject *args, PyObject *kwargs)
{
    const char *kwlist[] = {"top_left", "bottom_right", "zoom", NULL};
    double minLng, minLat, maxLng, maxLat;
    int zoom;

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "(dd)(dd)i", const_cast<char **>(kwlist), &minLng, &minLat, &maxLng, &maxLat, &zoom))
        return NULL;

    std::vector<Cluster*> clusters = self->sc->getClusters(
        std::make_pair(lngX(minLng), latY(minLat)),
        std::make_pair(lngX(maxLng), latY(maxLat)),
        zoom);

    PyObject *countKey = PyUnicode_FromString("count");
    PyObject *expansionZoomKey = PyUnicode_FromString("expansion_zoom");
    PyObject *idKey = PyUnicode_FromString("id");
    PyObject *latitudeKey = PyUnicode_FromString("latitude");
    PyObject *longitudeKey = PyUnicode_FromString("longitude");

    PyObject *o = NULL;
    PyObject *list = PyList_New(clusters.size());
    for (size_t i = 0; i < clusters.size(); ++i) {
        PyObject *dict = PyDict_New();
        Cluster *cluster = clusters[i];

        o = PyLong_FromSize_t(cluster->numPoints);
        PyDict_SetItem(dict, countKey, o);
        Py_DECREF(o);

        if (cluster->expansionZoom >= 0) {
            o = PyLong_FromSize_t(cluster->expansionZoom);
            PyDict_SetItem(dict, expansionZoomKey, o);
            Py_DECREF(o);
        } else {
            PyDict_SetItem(dict, expansionZoomKey, Py_None);
        }

        o = PyLong_FromSize_t(cluster->id);
        PyDict_SetItem(dict, idKey, o);
        Py_DECREF(o);

        o = PyFloat_FromDouble(yLat(cluster->point.second));
        PyDict_SetItem(dict, latitudeKey, o);
        Py_DECREF(o);

        o = PyFloat_FromDouble(xLng(cluster->point.first));
        PyDict_SetItem(dict, longitudeKey, o);
        Py_DECREF(o);

        PyList_SET_ITEM(list, i, dict);
    }

    Py_DECREF(countKey);
    Py_DECREF(expansionZoomKey);
    Py_DECREF(idKey);
    Py_DECREF(latitudeKey);
    Py_DECREF(longitudeKey);

    return list;
}


static PyMethodDef SuperCluster_methods[] = {
    {"getClusters", (PyCFunction)SuperCluster_getClusters, METH_VARARGS | METH_KEYWORDS, "Returns the clusters within the given bounding box at the given zoom level."},
    {NULL}
};


static PyTypeObject SuperClusterType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    "pysupercluster.SuperCluster",      /* tp_name */
    sizeof(SuperClusterObject),         /* tp_basicsize */
    0,                                  /* tp_itemsize */
    (destructor)SuperCluster_dealloc,   /* tp_dealloc */
    0,                                  /* tp_print */
    0,                                  /* tp_getattr */
    0,                                  /* tp_setattr */
    0,                                  /* tp_reserved */
    0,                                  /* tp_repr */
    0,                                  /* tp_as_number */
    0,                                  /* tp_as_sequence */
    0,                                  /* tp_as_mapping */
    0,                                  /* tp_hash  */
    0,                                  /* tp_call */
    0,                                  /* tp_str */
    0,                                  /* tp_getattro */
    0,                                  /* tp_setattro */
    0,                                  /* tp_as_buffer */
    Py_TPFLAGS_DEFAULT,                 /* tp_flags */
    "SuperCluster objects",             /* tp_doc */
    0,                                  /* tp_traverse */
    0,                                  /* tp_clear */
    0,                                  /* tp_richcompare */
    0,                                  /* tp_weaklistoffset */
    0,                                  /* tp_iter */
    0,                                  /* tp_iternext */
    SuperCluster_methods,               /* tp_methods */
    0,                                  /* tp_members */
    0,                                  /* tp_getset */
    0,                                  /* tp_base */
    0,                                  /* tp_dict */
    0,                                  /* tp_descr_get */
    0,                                  /* tp_descr_set */
    0,                                  /* tp_dictoffset */
    (initproc)SuperCluster_init,        /* tp_init */
    0,                                  /* tp_alloc */
};


static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "pysupercluster",                   /* m_name */
    "A fast geospatial point clustering module.",    /* m_doc */
    -1,                                 /* m_size */
    NULL,                               /* m_methods */
    NULL,                               /* m_reload */
    NULL,                               /* m_traverse */
    NULL,                               /* m_clear */
    NULL,                               /* m_free */
};


PyMODINIT_FUNC
PyInit_pysupercluster(void)
{
    PyObject* m;

    import_array();

    m = PyModule_Create(&moduledef);
    if (m == NULL)
        return NULL;

    SuperClusterType.tp_new = PyType_GenericNew;
    if (PyType_Ready(&SuperClusterType) < 0)
        return NULL;

    Py_INCREF(&SuperClusterType);
    PyModule_AddObject(m, "SuperCluster", (PyObject *)&SuperClusterType);

    return m;
}
