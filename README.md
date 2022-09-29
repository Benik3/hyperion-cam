# hyperion-cam
Simple camera grabber for Hyperion using Libcamera, OpenCV and flatbuffer.

## How it works
The app makes one stream from camera in full manual mode. There is no preview or any other output then to flatbuffer.  
Received frames can be transformed using OpenCv to correct picture deformation.  
Resulted frames are sent through flatbuffer connection to hyperion.

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
  -c [ --connection ] arg (=127.0.0.1) Socket address (IP or /tmp/hyperhdr-domain)  
  -p [ --port ] arg (=19400)           TCP port  
  -r [ --priority ] arg (=120)         Hyperion priority  
  -t [ --transform ] arg               Picture transformation in format   
                                       x1,y1:x2,y2:x3,y3:x4,y4  
  -v [ --verbose ]                     Verbose output  
  --help                               Print help  
```

#### Shutter note:
The shutter speed modifies the exposition time of the camera and FPS. The camera doesn't have fixed FPS, but the frames are released as soon as they are available. The value is in micro seconds.

#### Connection note:
The app supports using of TCP connection or Unix Domain Socket (available when using [HyperHDR](https://github.com/awawa-dev/HyperHDR) ). By specifying "/tmp/hyperhdr-domain" instead of IP automatically switch the app to IPC. IPC has advantage of faster data exchange with lower overhead.  
I don't recommend to use the TCP connection over a network as you will probably have lags (due to high packet size and packet framing). Maybe on ethernet network it would be usable, but over wifi it's unusable. Rather always install Hyperion locally and then send only the LED data to another Hyperion server (if you can't drive the LEDs directly).

#### Picture transformation note:
The app allows to make Perspective (projective) transformation of the picture in real time using OpenCV. Of course, it adds a little lag, but it's numbers of milliseconds.  
The transformation is described by 4 points in the original picture. The result is cropped and transformed so the 4 points make new corners of image.

## Credits
The camera part is based on [simple-cam](https://git.libcamera.org/libcamera/simple-cam.git) example from libcamera  
The OpenCV part is based on [OpenCV example](https://docs.opencv.org/3.4/da/d6e/tutorial_py_geometric_transformations.html)  
The Flatbuffer part is based on [Hyperion for LG WebOS](https://github.com/webosbrew/hyperion-webos) and [Hyperion-ng](https://github.com/hyperion-project/hyperion.ng).
