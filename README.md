# hyperion-cam
Simple camera grabber for Hyperion using Libcamera, OpenCV and flatbuffer.

## How it works
The app makes one stream from camera in full manual mode. There is no preview or any other output then to flatbuffer.  
Received frames can be transformed using OpenCv to correct picture deformation.  
Resulted frames are sent through flatbuffer connection to Hyperion/HyperHDR.

The grabber supports:
- RGB/BGR output (default)
- NV12 output via `--nv12`
- Optional GPU-accelerated transform (if compiled with EGL/GLES)

## How to use
By simply running without any options starts the app with these settings:  
640x480p, 60 FPS, Gain 1.0, WB gains 1.0, 1.0  
No transformation  
Connection to 127.0.0.1:19400, Origin name "camera", priority 120

Settings can be adjusted by using command line options:
```
  -w [ --width ] arg (=640)            Width  
  -h [ --height ] arg (=480)           Height  
  -s [ --shutter ] arg (=16666)        Shutter speed in us  
  -g [ --gain ] arg (=1)               Camera gain  
  -b [ --wbgain ] arg (=1.0,1.0)       White balance gains  
  -o [ --origin ] arg (=camera)        Name of the origin  
  -c [ --connection ] arg (=127.0.0.1) Socket address (IP or Unix socket path)  
  -p [ --port ] arg (=19400)           TCP port  
  -r [ --priority ] arg (=120)         Hyperion priority  
  -t [ --transform ] arg               Picture transformation in format   
                                       x1,y1:x2,y2:x3,y3:x4,y4  
  --imgout arg                         Capture one image (RGB) to file (png/bmp/...) and exit
  --nv12                               Enable NV12 format output (default is BGR888)
  --cpuonly                            Force CPU transform even if GPU is available
  -v [ --verbose ]                     Verbose output  
  --help                               Print help  
```

### One-shot image capture (`--imgout`)
Use `--imgout /path/file.png` to capture exactly one frame, save it, and exit.
- If `--transform` is provided, the saved image is the transformed output.
- `--imgout` always saves RGB/BGR images (there is no common still-image container for NV12). If `--nv12` is set, it is ignored.

Examples:
```
./hyperion-cam --imgout /tmp/frame.png
./hyperion-cam --transform "0,0:640,0:0,480:640,480" --imgout /tmp/frame.png
./hyperion-cam --cpuonly --transform "..." --imgout /tmp/frame.png
```

#### Shutter note:
The shutter speed modifies the exposition time of the camera and FPS. The camera doesn't have fixed FPS, but the frames are released as soon as they are available. The value is in micro seconds.

#### Connection note:
The app supports using of TCP connection or Unix Domain Socket (available when using [HyperHDR](https://github.com/awawa-dev/HyperHDR) at /tmp/hyperhdr-domain). If `--connection` is a valid IP address (IPv4/IPv6) it uses TCP (`--port` applies). Otherwise it treats the value as a Unix socket path. IPC has advantage of faster data exchange with lower overhead.  
I don't recommend to use the TCP connection over a network as you will probably have lags (due to high packet size and packet framing). Maybe on ethernet network it would be usable, but over wifi it's unusable. Rather always install Hyperion locally and then send only the LED data to another Hyperion server (if you can't drive the LEDs directly).

#### Picture transformation note:
The app allows to make Perspective (projective) transformation of the picture in real time using OpenCV. Of course, it adds a little lag, but it's numbers of milliseconds.  
The transformation is described by 4 points in the original picture. The result is cropped and transformed so the 4 points make new corners of image.

### GPU transform (optional)
If built with EGL + OpenGL ES support, GPU transform is used by default whenever `--transform` is provided.
Use `--cpuonly` to force CPU transforms.

### Performance / profiling knobs
These are optional environment variables used mainly for profiling and tuning:
- `HYPERCAM_CPU_PROFILE=1`: print average CPU transform time (RGB/NV12)
- `HYPERCAM_GL_PROFILE=1`: print average GPU stage timings
- `HYPERCAM_GL_PROFILE_SYNC=1`: GPU profiling with extra sync (more accurate, slower)
- `HYPERCAM_NV12_PBO=1`: NV12 readback pipeline using PBO+fences (can change latency/jitter)

GPU paths attempt DMA-BUF import automatically (zero-copy upload when supported) and fall back to CPU upload if not available.

## Credits
The camera part is based on [simple-cam](https://git.libcamera.org/libcamera/simple-cam.git) example from libcamera  
The OpenCV part is based on [OpenCV example](https://docs.opencv.org/3.4/da/d6e/tutorial_py_geometric_transformations.html)  
The Flatbuffer part is based on [Hyperion for LG WebOS](https://github.com/webosbrew/hyperion-webos) and [Hyperion-ng](https://github.com/hyperion-project/hyperion.ng).
