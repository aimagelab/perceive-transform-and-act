# Perceive, Transform, and Act
This is the PyTorch implementation of our paper:

**Perceive, Transform, and Act: Multi-Modal Attention Networks for Vision-and-Language Navigation**<br>
__***Federico Landi***__, Lorenzo Baraldi, Marcella Cornia, Massimiliano Corsini, Rita Cucchiara<br>

Our repository is based on the [Matterport3D simulator](https://github.com/peteanderson80/Matterport3DSimulator), which was originally proposed with the Room-to-Room dataset.

## Installation

As a first step, clone the repository and create the environment with conda:

```
git clone --recursive https://github.com/aimagelab/perceive-transform-and-act
cd perceive-transform-and-act

conda env create -f environment.yml
source activate pta
```

If you didn't clone with the `--recursive` flag, then you'll need to manually clone the submodules from the top-level directory:
```
git submodule update --init --recursive
```

### Building with Docker

Please follow the instructions on the [Matterport3DSimulator repository](https://github.com/peteanderson80/Matterport3DSimulator) to install the simulator via Docker.

### Bulding without Docker

A C++ compiler with C++11 support is required. Matterport3D Simulator has several dependencies:
- Ubuntu >= 14.04
- Nvidia-driver with CUDA installed 
- C++ compiler with C++11 support
- [CMake](https://cmake.org/) >= 3.10
- [OpenCV](http://opencv.org/) >= 2.4 including 3.x
- [OpenGL](https://www.opengl.org/)
- [GLM](https://glm.g-truc.net/0.9.8/index.html)
- [Numpy](http://www.numpy.org/)

Optional dependences (depending on the cmake rendering options):
- [OSMesa](https://www.mesa3d.org/osmesa.html) for OSMesa backend support
- [epoxy](https://github.com/anholt/libepoxy) for EGL backend support

If all of the dependecies are installed, you can build the simulator from source by tiping:

```
mkdir build
cd build
cmake -DOSMESA_RENDERING=ON -DPYTHON_EXECUTABLE:FILEPATH=`path/to/your/python/bin` ..
make
```

### Precomputed ResNet Image Features

Download the [precomputed ResNet-152 (imagenet) features](https://www.dropbox.com/s/715bbj8yjz32ekf/ResNet-152-imagenet.zip?dl=1), and place the corresponding .tsv file into the ```img_features``` folder.

## Training and Testing

To train PTA from scratch, move to the root directory and run:

```
python tasks/R2R/main.py --name train_from_scratch \
                         --plateau_sched \
                         --lr 1e-4 \
                         --max_episode_len 30
```

We also provide weights obtained with the training described in the paper. If you wish to reproduce the results in our paper, run:

```
python tasks/R2R/main.py --name test_ll \
                         --max_episode_len 30 \
                         --eval_only \
                         --pretrained \
                         --load_from low_level
```

Our agent can also perform high-level Vision-and-Language Navigation.
To reproduce the results otained with the high-level setup, run:

```
python tasks/R2R/main.py --name test_hl \
                         --high_level \
                         --max_episode_len 10 \
                         --eval_only \
                         --pretrained \
                         --load_from high_level
```

## Visualizing Navigation Episodes

To make our qualitative results easier to visualize, we provide some .gif files that display some of the navigation episodes reported in our paper. We also show meaningful metrics to evaluate our results.

### Low-level VLN in R2R

<p>
<img src="teaser/r2r_3.gif" width="420">
  &nbsp; &nbsp;
<img src="teaser/r2r_2.gif" width="420">
</p>

### Low-level VLN in R4R

<p>
<img src="teaser/r4r_1.gif" width="420">
  &nbsp; &nbsp;
<img src="teaser/r4r_2.gif" width="420">
</p>

## Reproducibility Note

Our experiments were made using an Nvidia 1080Ti GPU, CUDA 10.0, and python 3.6.8. Using different hardware setups or software versions may affect results.
