#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <Python.h>
#include <numpy/arrayobject.h>

#include "supercluster.hpp"


typedef struct {
    PyObject_HEAD
    SuperCluster *sc;
} SuperClusterObject;


static int
SuperCluster_init(SuperClusterObject *self, PyObject *args)
{
    PyArrayObject *array;
    if (!PyArg_ParseTuple(args, "O!", &PyArray_Type, &array))
        return -1;

    if (PyArray_DESCR(array)->type_num != NPY_DOUBLE || PyArray_NDIM(array) != 2) {
        PyErr_SetString(PyExc_ValueError, "Array must be of type double and 2 dimensional.");
        return -1;
    }

    npy_intp count = PyArray_DIMS(array)[0];
    std::vector<Point> items(count);
    for (npy_intp i = 0; i < count; ++i) {
        items[i] = std::make_pair(
            *(double*)PyArray_GETPTR2(array, i, 0),
            *(double*)PyArray_GETPTR2(array, i, 1));
    }
    self->sc = new SuperCluster(items);

    return 0;
}


static void
SuperCluster_dealloc(SuperClusterObject *self)
{
    delete self->sc;
}

static PyMethodDef SuperCluster_methods[] = {
    {NULL}
};

static PyTypeObject SuperClusterType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    "supercluster.SuperCluster",        /* tp_name */
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
    "supercluster",                     /* m_name */
    "A crazy fast geospatial point clustering.",    /* m_doc */
    -1,                                 /* m_size */
    NULL,                               /* m_methods */
    NULL,                               /* m_reload */
    NULL,                               /* m_traverse */
    NULL,                               /* m_clear */
    NULL,                               /* m_free */
};


PyMODINIT_FUNC
PyInit_supercluster(void)
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
