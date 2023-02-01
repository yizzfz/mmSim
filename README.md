# mmSim

A simulator of mmWave radar that simulates the IF signal as if the radar is placed in a customized scene, for algorithm design and verification.

It supports configuration of:
* Number and layout of receiver array. It has several built-in layouts for Texas Instruments radars.
* Chirp configuration, such as slope, duration, frequency, number of chirps per frame, ADC sampling rate, etc.
* Object in the scene, represented as points. Support 3D model import from some [pyTorch geometric datasets](https://pytorch-geometric.readthedocs.io/en/latest/modules/datasets.html).
* Object motion. 
* Signal-to-noise ratio.

The project contains algorithms for detecting the range, velocity, and angle-of-arrival of the objects and estimating a point cloud based on the simulation data.

## Prerequisite 

Install Python (tested with 3.9) and install dependencies with ```pip install -r requirement.txt```.

## Demos 

The `demo-simple.py` script demonstrates the basic functions of the tool. It sets up a scene consisting of a single point with a motion, and shows how the simulation data can be interpreted. 

```python demo-simple.py```

The `demo-pointcloud-eval.py` script demonstrates how to import a human model from the [FAUST](http://faust.is.tue.mpg.de/) dataset, get the simulation data, and estimate a point cloud based using the data.

```python demo-pointcloud-eval.py```

## Citation

If you find this work useful, please consider cite:
```
@misc{Cui23,
  doi = {10.48550/ARXIV.2301.13553},
  url = {https://arxiv.org/abs/2301.13553},
  author = {Cui, Han and Wu, Jiacheng and Dahnoun, Naim},
  title = {Millimetre-wave Radar for Low-Cost 3D Imaging: A Performance Study},
  publisher = {arXiv},
  year = {2023}
}
```
