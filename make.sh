rm ./pylibtim/libtim.cpython*
c++ -I./libtim/. -O3 -shared -std=c++11 -fPIC $(python3 -m pybind11 --includes) libtim.cpp -o ./pylibtim/libtim$(python3-config --extension-suffix)
