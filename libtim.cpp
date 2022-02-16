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
enum AttributeID {
  AREA,
  MSER,
  CONTRAST,
  VOLUME,
  CONTOUR_LENGTH,
  COMPLEXITY,
  COMPACITY,
  HU_INVARIANT_MOMENT
};
enum AttributeValID { NODE, MAX_PARENTS, MIN_PARENTS };

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

long getAttribute(Node *n, AttributeID attribute_id) {
  switch (attribute_id) {
    case AREA:
      return (long)n->area;
    case MSER:
      return (long)(n->mser*10000L);
    case CONTRAST:
      return (long)n->contrast;
    case VOLUME:
      return (long)n->volume;
    case CONTOUR_LENGTH:
      return (long)n->contourLength;
    case COMPLEXITY:
      return (long)n->complexity;
    case COMPACITY:
      return (long)n->compacity;
    case HU_INVARIANT_MOMENT:
        return (long)n->I;
  }
  return 0;
}

void area_filtering(int x, int y, int z, py::array_t<uint8_t> image, int area,
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
  tree.areaFiltering(area);

  // Reconstruction d'une image à partir de l'arbre élagué
  Image<U8> res = tree.constructImage(ComponentTree<U8>::DIRECT);

  for (py::ssize_t i = 0; i < r.shape(0); i++)
    for (py::ssize_t j = 0; j < r.shape(1); j++)
      for (py::ssize_t k = 0; k < r.shape(2); k++)
        r(i, j, k) = (uint8_t)res(i, j, k);
}

void attribute_image(int x, int y, int z, py::array_t<uint8_t> image,
                     py::array_t<long> image_attr, ConnexityID connexity_id,
                     AttributeID attribute_id,
                     AttributeValID attribute_val_id) {
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
  ComponentTree<U8> tree(im, connexity);

  // Reconstruction d'une image à partir de la valeur des attributs
  // construction de la liste des noeuds
  std::vector<Node *> nodes = tree.indexedNodes();
  for (py::ssize_t i = 0; i < r.shape(0); i++)
    for (py::ssize_t j = 0; j < r.shape(1); j++)
      for (py::ssize_t k = 0; k < r.shape(2); k++) {
        Node *n = tree.indexedCoordToNode(i, j, k, nodes);
        long attr = getAttribute(n, attribute_id);

        if (attribute_val_id == NODE) {
          // nothing.
        } else if (attribute_val_id == MAX_PARENTS) {
          while (n->father != tree.m_root) {
            n = n->father;
            attr = std::max(attr, getAttribute(n, attribute_id));
          }
          attr = std::max(attr, getAttribute(n, attribute_id));
        } else if (attribute_val_id == MIN_PARENTS) {
          while (n->father != tree.m_root) {
            n = n->father;
            attr = std::min(attr, getAttribute(n, attribute_id));
          }
          attr = std::min(attr, getAttribute(n, attribute_id));
        }
        r_attr(i, j, k) = attr;
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

  py::enum_<AttributeID>(m, "AttributeID")
      .value("AREA", AREA)
      .value("MSER", MSER)
      .value("CONTRAST", CONTRAST)
      .value("VOLUME", VOLUME)
      .value("CONTOUR_LENGTH", CONTOUR_LENGTH)
      .value("COMPLEXITY", COMPLEXITY)
      .value("COMPACITY", COMPACITY)
      .value("HU_INVARIANT_MOMENT", HU_INVARIANT_MOMENT)
      .export_values();

  py::enum_<AttributeValID>(m, "AttributeValID")
      .value("NODE", NODE)
      .value("MAX_PARENTS", MAX_PARENTS)
      .value("MIN_PARENTS", MIN_PARENTS)
      .export_values();
}
