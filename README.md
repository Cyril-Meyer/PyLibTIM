# PyLibTIM

[PyLibTIM](https://github.com/Cyril-Meyer/PyLibTIM) is a python library that binds some features of LibTIM.  
[LibTIM](https://github.com/bnaegel/libtim) is an image processing library in C++ under GPL 3.0 License.  
PyLibTIM does not provide a full binding of LibTIM functions, but rather a collection of ready-to-use functions based on LibTIM.

#### Setup guide
*Create venv*
```
python -m venv venv
source venv/bin/activate
pip install -U pip
```
*Install requirements*
```
pip install numpy
pip install "pybind11[global]"
```
*make PyLibTIM*
```
./make.sh
```

Now you can copy the pylibtim folder to the projects where you want to use it.
You can also copy it to a folder like `/usr/local/lib` and create a symbolic link `ln -s /usr/local/lib/pylibtim pylibtim`

*test.py (examples) requirements*
```
pip install matplotlib
pip install tifffile
```
