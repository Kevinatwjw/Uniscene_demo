# Installation

> We recommend creating an environment using the following script with "poetry", which has been tested on NVIDIA A100/A800 with CUDA 12.1, Python 3.9, under Ubuntu 20.04.

a. Install CUDA

- Install CUDA=12.1 and cudnn.

- Set environment variables.

  For example

  ```bash
  CUDA=12.1
  CUDNN=8.8.1.3

  export CUDA_HOME=/data/cuda/cuda-$CUDA/cuda
  export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$CUDA_HOME/lib64/stubs:$CUDA_HOME/extras/CUPTI/lib64:/data/cuda/cuda-$CUDA/cudnn/v$CUDNN/lib64:/usr/local/nvidia/lib64:$LD_LIBRARY_PATH
  export PATH=$CUDA_HOME/bin:$PATH
  export CUDA_TOOLKIT_ROOT_DIR=$CUDA_HOME
  export LIBRARY_PATH=$CUDA_HOME/lib64:$LIBRARY_PATH
  export CFLAGS="-I$CUDA_HOME/include $CFLAGS"
  export LDFLAGS="-L/usr/local/nvidia/lib64 $CFLAGS"
  ```

b. GCC: Make sure gcc is >= 9 in the conda environment.

> We have tested with gcc 9.4.0.

c. Create a conda virtual environment

```bash
git clone git@github.com:Arlo0o/UniScene-Unified-Occupancy-centric-Driving-Scene-Generation.git --recursive

# install system dependency
sudo apt-get install xvfb

conda create -n uniscene python=3.9
conda activate uniscene
conda install pytorch=2.5.1=py3.9_cuda12.1_cudnn9.1.0_0 torchvision  -c pytorch -c nvidia

pip install -U openmim
mim install mmcv==2.1.0

# for mayavi
pip install --no-cache-dir setuptools==65.5.0
pip install poetry
poetry install --no-root

# build
cd ./lidar_gen/ && pip install -e . -v
cd ../
cd ./video_gen/gs_render/diff-gaussian-rasterization/  && pip install ./
cd ../../../

cd third_party/chamferdist
python3 setup.py install
cd ../../

export PYTHONPATH=$(pwd)
```

d. (Optional) Visualization by mayavi

> We have found that `mayavi` may cause environmental conflicts or runtime core dump issues. Here we provide a possible solution that may be applicable to you. If you have any questions about this part, you can seek help or find solutions [here](https://github.com/enthought/mayavi/issues).

- Install some system packages.

  ```bash
  sudo apt install libxcb-cursor0
  sudo apt install libxcb-cursor-dev
  sudo apt-get install libxcb-xinerama0
  ```

- Install `mayavi`, `PyQt5`, `vtk`, `setuptools` if you failed to install them by `poetry`.
  Check `mayavi`, `PyQt5`, `vtk`, `setuptools` version by `pip list`.

  ```bash
  pip install --no-cache-dir setuptools==65.5.0
  pip install vtk==9.0.2 PyQt5==5.15.10 PyQt5-sip
  pip install --no-cache-dir mayavi==4.7.3
  ```

- Rendering using the virtual framebuffer as suggested by [here](https://docs.enthought.com/mayavi/mayavi/tips.html#rendering-using-the-virtual-framebuffer)

  ```bash
  # Run this command in another terminal.
  Xvfb :1 -screen 0 1280x1024x24 -auth localhost
  ```

  ```bash
  # Run this command before visualization or other python scripts.
  export DISPLAY=:1
  export QT_QPA_PLATFORM=offscreen
  export ETS_TOOLKIT=qt
  export QT_API=pyqt5
  # If mayavi is installed correctly, it should generate and save an image named mayavi_test.png here.
  python3 occupancy_gen/check_mayavi.py

  # cd occupancy_gen
  # bash run_eval_dit.sh
  ```
