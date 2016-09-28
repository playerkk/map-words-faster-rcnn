### About

This repository contains Python testing code for words detection, which (including this README) is based on the awesome [py-faster-rcnn repository](https://github.com/rbgirshick/py-faster-rcnn). 

### License

It is released under the MIT License (as of py-faster-rcnn. Refer to the LICENSE file for details).

### Contents
1. [Requirements: software](#requirements-software)
2. [Requirements: hardware](#requirements-hardware)
3. [Basic installation](#installation-sufficient-for-the-demo)
4. [Demo](#demo)

### Requirements: software

1. Requirements for `Caffe` and `pycaffe` (see: [Caffe installation instructions](http://caffe.berkeleyvision.org/installation.html))

  **Note:** Caffe *must* be built with support for Python layers!

  ```make
  # In your Makefile.config, make sure to have this line uncommented
  WITH_PYTHON_LAYER := 1
  # Unrelatedly, it's also recommended that you use CUDNN
  USE_CUDNN := 1
  ```

  You can download my [Makefile.config](http://www.cs.berkeley.edu/~rbg/fast-rcnn-data/Makefile.config) for reference.
2. Python packages you might not have: `cython`, `python-opencv`, `easydict`

### Requirements: hardware

1. It is strongly recommened to use a morden GPU (e.g., Titan or K40).

### Installation (sufficient for the demo)

1. Clone the repository
  ```Shell
  # Make sure to clone with --recursive
  git clone https://github.com/playerkk/map-words-faster-rcnn.git
  ```

2. Build the Cython modules
    # We'll call the directory that you cloned this repository `FRCN_ROOT`. 
    ```Shell
    cd $FRCN_ROOT/lib
    make
    ```

3. Build Caffe and pycaffe
    ```Shell
    cd $FRCN_ROOT/caffe-fast-rcnn
    # Now follow the Caffe installation instructions here:
    #   http://caffe.berkeleyvision.org/installation.html

    # If you're experienced with Caffe and have all of the requirements installed
    # and your Makefile.config in place, then simply do:
    make -j8 && make pycaffe
    ```

4. Download pre-computed Faster R-CNN detectors
    ```Shell
    cd $FRCN_ROOT
    ./pre-trained-models/fetch_pre_trained_models.sh
    ```

    This will download a pre-trained words detection model into the `pre-trained-models' folder.

### Demo

*After successfully completing [basic installation](#installation-sufficient-for-the-demo)*, you'll be ready to run the demo.

To run the demo
```Shell
cd $FRCN_ROOT
./tools/demo.py
```
The demo performs words detection using a pre-trained VGG16 network.