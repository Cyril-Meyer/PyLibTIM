#include <ctime>
#include <iostream>

#include "libtim/Algorithms/ComponentTree.h"
#include "libtim/Algorithms/ConnectedComponents.h"
#include "libtim/Algorithms/Thresholding.h"
#include "libtim/Common/FlatSE.h"
#include "libtim/Common/Image.h"

using namespace std;
using namespace LibTIM;

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

enum ConnexityID { C2DN4, C2DN8, C3DN6, C3DN18, C3DN26 };

FlatSE getFlatSE(int connexity_id) {
  FlatSE connexity;
  switch (connexity_id) {
    case C2DN4:
      connexity.make2DN4();
      break;
    case C2DN8:
      connexity.make2DN8();
      break;
    case C3DN6:
      connexity.make3DN6();
      break;
    case C3DN18:
      connexity.make3DN18();
      break;
    case C3DN26:
      connexity.make3DN26();
      break;
    default:
      connexity.make2DN8();
      break;
  }
  return connexity;
}

int32_t int64Toint32(int64_t a)
{
    return (int32_t)std::max(std::min(a, (int64_t)std::numeric_limits<int32_t>::max()), (int64_t)std::numeric_limits<int32_t>::min());
}

void area_filtering(int x, int y, int z, py::array_t<uint8_t> image, int area_min, int area_max,
                    int connexity_id) {
  auto r = image.mutable_unchecked<3>();

  // Image
  // TSize size[] = {x, y, z};
  // TSpacing spacing[] = {1.0, 1.0, 1.0};
  // const U8 data[] = {0, 1, 2, 3, 4, ...}
  TSize size[] = {(short unsigned int)x, (short unsigned int)y,
                  (short unsigned int)z};
  Image<U8> im(size);

  for (py::ssize_t i = 0; i < r.shape(0); i++)
    for (py::ssize_t j = 0; j < r.shape(1); j++)
      for (py::ssize_t k = 0; k < r.shape(2); k++) im(i, j, k) = r(i, j, k);

  // Construction du component-tree
  FlatSE connexity = getFlatSE(connexity_id);
  ComponentTree<U8> tree(im, connexity);

  // Elagage de l'arbre
  tree.areaFiltering(area_min, area_max);

  // Reconstruction d'une image à partir de l'arbre élagué
  Image<U8> res = tree.constructImage(ComponentTree<U8>::DIRECT);

  for (py::ssize_t i = 0; i < r.shape(0); i++)
    for (py::ssize_t j = 0; j < r.shape(1); j++)
      for (py::ssize_t k = 0; k < r.shape(2); k++)
        r(i, j, k) = (uint8_t)res(i, j, k);
}

void attribute_image(int x, int y, int z, py::array_t<uint8_t> image,
                     py::array_t<long double> image_attr, ConnexityID connexity_id,
                     ComponentTree<U8>::Attribute value_attribute,
                     ComponentTree<U8>::Attribute selection_attribute, unsigned int delta,
                     ComponentTree<U8>::ConstructionDecision selection_rule=ComponentTree<U8>::DIRECT) {
  auto r = image.mutable_unchecked<3>();
  auto r_attr = image_attr.mutable_unchecked<3>();

  // Image
  // TSize size[] = {x, y, z};
  // TSpacing spacing[] = {1.0, 1.0, 1.0};
  // const U8 data[] = {0, 1, 2, 3, 4, ...}
  TSize size[] = {(short unsigned int)x, (short unsigned int)y,
                  (short unsigned int)z};
  Image<U8> im(size);

  for (py::ssize_t i = 0; i < r.shape(0); i++)
    for (py::ssize_t j = 0; j < r.shape(1); j++)
      for (py::ssize_t k = 0; k < r.shape(2); k++) im(i, j, k) = r(i, j, k);

  // Construction du component-tree
  FlatSE connexity = getFlatSE(connexity_id);
  ComponentTree<U8> tree(im, connexity, delta);

  // Reconstruction d'une image à partir de la valeur des attributs
  // todo : dynamic type ? typeid() ?
  Image<int64_t> res = tree.constructImageAttribute<int64_t, long double>(value_attribute, selection_attribute, selection_rule);

  for (py::ssize_t i = 0; i < r.shape(0); i++)
    for (py::ssize_t j = 0; j < r.shape(1); j++)
      for (py::ssize_t k = 0; k < r.shape(2); k++) {
        r_attr(i, j, k) = (long double)res(i, j, k);;
      }
}

PYBIND11_MODULE(libtim, m) {
  m.doc() = "libtim";
  m.def("area_filtering", &area_filtering);
  m.def("attribute_image", &attribute_image);

  py::enum_<ConnexityID>(m, "ConnexityID")
      .value("C2DN4", C2DN4)
      .value("C2DN8", C2DN8)
      .value("C3DN6", C3DN6)
      .value("C3DN18", C3DN18)
      .value("C3DN26", C3DN26)
      .export_values();

  py::enum_<ComponentTree<U8>::Attribute>(m, "AttributeID")
      .value("H", ComponentTree<U8>::H)
      .value("AREA", ComponentTree<U8>::AREA)
      .value("AREA_D_AREAN_H", ComponentTree<U8>::AREA_D_AREAN_H)
      .value("AREA_D_AREAN_H_D", ComponentTree<U8>::AREA_D_AREAN_H_D)
      .value("AREA_D_H", ComponentTree<U8>::AREA_D_H)
      .value("AREA_D_AREAN", ComponentTree<U8>::AREA_D_AREAN)
      .value("MSER", ComponentTree<U8>::MSER)
      .value("AREA_D_DELTA_H", ComponentTree<U8>::AREA_D_DELTA_H)
      .value("AREA_D_DELTA_AREAF", ComponentTree<U8>::AREA_D_DELTA_AREAF)
      .value("MEAN", ComponentTree<U8>::MEAN)
      .value("VARIANCE", ComponentTree<U8>::VARIANCE)
      .value("MEAN_NGHB", ComponentTree<U8>::MEAN_NGHB)
      .value("VARIANCE_NGHB", ComponentTree<U8>::VARIANCE_NGHB)
      .value("OTSU", ComponentTree<U8>::OTSU)
      .value("CONTRAST", ComponentTree<U8>::CONTRAST)
      .value("VOLUME", ComponentTree<U8>::VOLUME)
      .value("MGB", ComponentTree<U8>::MGB)
      .value("CONTOUR_LENGTH", ComponentTree<U8>::CONTOUR_LENGTH)
      .value("COMPLEXITY", ComponentTree<U8>::COMPLEXITY)
      .value("COMPACITY", ComponentTree<U8>::COMPACITY)
      .export_values();

  py::enum_<ComponentTree<U8>::ConstructionDecision>(m, "ConstructionDecision")
      .value("DIRECT", ComponentTree<U8>::DIRECT)
      .value("MIN", ComponentTree<U8>::MIN)
      .value("MAX", ComponentTree<U8>::MAX)
      .export_values();
}
