#include <iostream>
#include <ctime>

#include "libtim/Common/Image.h"
#include "libtim/Common/FlatSE.h"
#include "libtim/Algorithms/ComponentTree.h"
#include "libtim/Algorithms/ConnectedComponents.h"
#include "libtim/Algorithms/Thresholding.h"

using namespace std;
using namespace LibTIM;

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

FlatSE getFlatSE(int connexity_id)
{
	FlatSE connexity;
	switch(connexity_id)
	{
		case 1:
		connexity.make2DN4();
		break;
		case 2:
		connexity.make2DN8();
		break;
		case 3:
		connexity.make3DN6();
		break;
		case 4:
		connexity.make3DN18();
		break;
		case 5:
		connexity.make3DN26();
		break;
		default:
		connexity.make2DN8();
		break;
	}
	return connexity;
}

void area_filtering(int x, int y, int z, py::array_t<uint8_t> image, int area, int connexity_id)
{
	auto r = image.mutable_unchecked<3>();

  // Image
  // TSize size[] = {x, y, z};
  // TSpacing spacing[] = {1.0, 1.0, 1.0};
  // const U8 data[] = {0, 1, 2, 3, 4, ...}
	TSize size[] = {(short unsigned int)x,
								  (short unsigned int)y,
									(short unsigned int)z};
  Image<U8> im(size);

	for (py::ssize_t i = 0; i < r.shape(0); i++)
			for (py::ssize_t j = 0; j < r.shape(1); j++)
					for (py::ssize_t k = 0; k < r.shape(2); k++)
							im(i, j, k) = r(i, j, k);

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

PYBIND11_MODULE(libtim, m)
{
  m.doc() = "libtim";
  m.def("area_filtering", &area_filtering);
}
