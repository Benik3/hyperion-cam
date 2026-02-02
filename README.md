# hyperion-cam
Simple camera grabber for Hyperion using Libcamera, EGL/OpenCV and flatbuffer.

## How it works
The app makes one stream from camera in full manual mode. There is no preview or any other output than to flatbuffer.  
Received frames can be transformed to correct perspective picture deformation.  
Resulted frames are sent through flatbuffer connection to Hyperion/HyperHDR.

The grabber supports:
- RGB/BGR output (default)
- NV12 output via `--nv12`
- Optional GPU-accelerated transform (if compiled with EGL/GLES)

## How to build
This project uses CMake and pkg-config.

Dependencies (Debian/Ubuntu/Raspberry Pi OS):
```
sudo apt update
sudo apt install -y \
  build-essential cmake pkg-config \
  libcamera-dev libevent-dev libboost-program-options-dev libopencv-dev
```

Optional but recommended (GPU transform support):
```
sudo apt install -y libegl1-mesa-dev libgles2-mesa-dev
```

Build:
```
mkdir -p Release
cd Release
cmake ..
make
```

The binary will be at `Release/hyperion-cam`.

## How to use
For the usage and settings check the wiki.

## Credits
The camera part is based on [simple-cam](https://git.libcamera.org/libcamera/simple-cam.git) example from libcamera  
The OpenCV part is based on [OpenCV example](https://docs.opencv.org/3.4/da/d6e/tutorial_py_geometric_transformations.html)  
The Flatbuffer part is based on [Hyperion for LG WebOS](https://github.com/webosbrew/hyperion-webos) and [HyperHDR](https://github.com/awawa-dev/HyperHDR)
