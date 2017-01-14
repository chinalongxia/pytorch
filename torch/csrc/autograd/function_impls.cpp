#include <Python.h>
#include <structmember.h>

#include <unordered_map>
#include <unordered_set>
#include <exception>

#include "THP.h"

#define MAKE_TYPE_OBJECT(FN_NAME, BASE_CLS, MEMBERS, METHODS, INIT_FN)         \
PyTypeObject THP##FN_NAME##Type = {                                            \
  PyVarObject_HEAD_INIT(NULL, 0)                                               \
  "torch._C._" #FN_NAME "Base",          /* tp_name */                         \
  sizeof(THPFunction),                   /* tp_basicsize */                    \
  0,                                     /* tp_itemsize */                     \
  0,                                     /* tp_dealloc */                      \
  0,                                     /* tp_print */                        \
  0,                                     /* tp_getattr */                      \
  0,                                     /* tp_setattr */                      \
  0,                                     /* tp_reserved */                     \
  0,                                     /* tp_repr */                         \
  0,                                     /* tp_as_number */                    \
  0,                                     /* tp_as_sequence */                  \
  0,                                     /* tp_as_mapping */                   \
  0,                                     /* tp_hash  */                        \
  0,                                     /* tp_call */                         \
  0,                                     /* tp_str */                          \
  0,                                     /* tp_getattro */                     \
  0,                                     /* tp_setattro */                     \
  0,                                     /* tp_as_buffer */                    \
  Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE | Py_TPFLAGS_HAVE_GC, /* tp_flags */\
  NULL,                                  /* tp_doc */                          \
  0,                                     /* tp_traverse */                     \
  0,                                     /* tp_clear */                        \
  0,                                     /* tp_richcompare */                  \
  0,                                     /* tp_weaklistoffset */               \
  0,                                     /* tp_iter */                         \
  0,                                     /* tp_iternext */                     \
  METHODS,                               /* tp_methods */                      \
  MEMBERS,                               /* tp_members */                      \
  0,                                     /* tp_getset */                       \
  &THP##BASE_CLS##Type,                  /* tp_base */                         \
  0,                                     /* tp_dict */                         \
  0,                                     /* tp_descr_get */                    \
  0,                                     /* tp_descr_set */                    \
  0,                                     /* tp_dictoffset */                   \
  (initproc)INIT_FN,                     /* tp_init */                         \
  0,                                     /* tp_alloc */                        \
  0                                      /* tp_new */                          \
};


////////////////////////////////////////////////////////////////////////////////
// InplaceFunction
////////////////////////////////////////////////////////////////////////////////

int THPInplaceFunction_init(THPFunction *self, PyObject *args, PyObject *kwargs)
{
  int num_args = PyTuple_GET_SIZE(args);
  int num_kwargs = kwargs ? PyDict_Size(kwargs) : 0;
  int total_args = num_args + num_kwargs;
  if (total_args > 1) {
    PyErr_SetString(PyExc_TypeError, "InplaceFunction __init__ takes only "
        "one single argument");
    return -1;
  }

  PyObject *arg = NULL;
  if (num_args > 0) {
    arg = PyTuple_GET_ITEM(args, 0);
  } else if (num_kwargs > 0) {
    arg = PyDict_GetItemString(kwargs, "inplace");
    if (arg == NULL) {
      PyErr_Format(PyExc_TypeError, "InplaceFunction __init__ got an invalid "
          "keyword argument");
      return -1;
    }
  }

  if (arg) {
    if (!PyBool_Check(arg)) {
      PyErr_Format(PyExc_TypeError, "InplaceFunction __init__ accepts a single "
          "bool but got %s", THPUtils_typename(arg));
      return -1;
    }
    self->inplace = arg == Py_True;
  }

  return 0;
}


static struct PyMemberDef THPInplaceFunction_members[] = {
  {(char*)"inplace", T_BOOL, offsetof(THPFunction, inplace), 0, NULL},
  {NULL}
};

MAKE_TYPE_OBJECT(InplaceFunction, Function, THPInplaceFunction_members, NULL,
    THPInplaceFunction_init);

////////////////////////////////////////////////////////////////////////////////
// BlasBase
////////////////////////////////////////////////////////////////////////////////

// WARNING: this *DOES NOT* call the __init__ of superclass
static int THPBlasFunction_init(THPFunction *self, PyObject *args, PyObject *kwargs)
{
  static const char *arg_names[] = {"alpha", "beta", "inplace"};
  static PyObject *one = PyInt_FromLong(1);
  PyObject *alpha = one;
  PyObject *beta = one;
  PyObject *inplace = Py_False;
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OOO", (char**)arg_names,
        &alpha, &beta, &inplace))
    return -1;

  if (!PyBool_Check(inplace)) {
    PyErr_Format(PyExc_TypeError, "inplace argument is expected to be a bool, "
        "but got %s", THPUtils_typename(inplace));
    return -1;
  }
  self->inplace = inplace == Py_True;

  if (PyObject_SetAttrString((PyObject*)self, "alpha", alpha) == -1) return -1;
  if (PyObject_SetAttrString((PyObject*)self, "beta", beta) == -1) return -1;

  return 0;
}

static PyObject* THPBlasFunction_get_output(THPFunction *self, PyObject *arg)
{
  //THPUtils_assert(THPModule_isTensor(arg), "_get_output argument has to be a "
      //"tensor, but got %s", THPUtils_typename(arg));
  if (self->inplace) {
    self->dirty_tensors = PyTuple_New(1);
    if (!self->dirty_tensors) return NULL;
    Py_INCREF(arg);
    PyTuple_SET_ITEM(self->dirty_tensors, 0, arg);
    Py_INCREF(arg);
    return arg;
  } else {
    THPObjectPtr new_arg = PyObject_CallMethod(arg, "new", "");
    if (!new_arg) return NULL;
    return PyObject_CallMethod(new_arg, "resize_as_", "O", arg);
  }
}

static struct PyMethodDef THPBlasFunction_methods[] = {
  {(char*)"_get_output", (PyCFunction)THPBlasFunction_get_output, METH_O, NULL},
  {NULL}
};


MAKE_TYPE_OBJECT(BlasFunction, InplaceFunction, NULL,
    THPBlasFunction_methods, THPBlasFunction_init);

////////////////////////////////////////////////////////////////////////////////
// Module init
////////////////////////////////////////////////////////////////////////////////

static bool _register_fn(PyObject *module, PyTypeObject *type, const char *name)
{
  if (PyType_Ready(type) < 0) return false;
  Py_INCREF(type);
  PyModule_AddObject(module, name, (PyObject *)type);
  return true;
}

bool THPFunctionImpls_initModule(PyObject *module)
{
  if (!_register_fn(module, &THPInplaceFunctionType, "_InplaceFunctionBase"))
    return false;
  if (!_register_fn(module, &THPBlasFunctionType, "_BlasFunctionBase"))
    return false;
  return true;
}
