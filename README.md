# hyperion-cam
Simple camera grabber for Hyperion using Libcamera, OpenCV and flatbuffer.

## How it works
The app makes one stream from camera in full manual mode. There is no preview or any other output than to flatbuffer.  
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
  --fps arg (=0)                       Target FPS limit (0=disabled, adds delay to maintain frame rate)
  --max-pending-process arg (=0)       Max in-flight transform jobs (0=unlimited). Set to 1 for lowest latency (drops frames when busy)
  --max-pending-send arg (=0)          Max in-flight send jobs (0=unlimited). Set to 1-2 for lowest latency (drops frames when busy)
  -g [ --gain ] arg (=1)               Camera gain  
  -b [ --wbgain ] arg (=1.0,1.0)       White balance gains  
  -f [ --focus ] arg (=-1)             Manual focus lens position (-1 keeps default)
  --saturation arg (=1)                Camera saturation
  --contrast arg (=1)                  Camera contrast
  --gamma arg (=0)                     Camera gamma (0.0=use camera default)
  --hflip                              Horizontal flip (native camera transform)
  --vflip                              Vertical flip (native camera transform)
  --rotate arg (=0)                    Rotation in degrees: 0, 90, 180, 270 (native camera transform)
  --crop arg                           Sensor-level crop in format x,y,width,height (absolute sensor coordinates)
  -o [ --origin ] arg (=camera)        Name of the origin  
  -c [ --connection ] arg (=127.0.0.1) Socket address (IP or Unix socket path)  
  -p [ --port ] arg (=19400)           TCP port  
  -r [ --priority ] arg (=120)         Hyperion priority  
  -t [ --transform ] arg               Picture transformation in format   
                                       x1,y1:x2,y2:x3,y3:x4,y4  
  --output-crop arg                    Output crop in format x,y,width,height (applied after transform)
  --imgout arg                         Capture one image (RGB) to file (png/bmp/...) and exit
  --imgout-crophelp arg                Capture image showing sensor area used at requested resolution (helps determine --crop coordinates)
  --nv12                               Enable NV12 format output (default is BGR888)
  --cpuonly                            Force CPU transform even if GPU is available
  -v [ --verbose ]                     Verbose output  
  --version                            Print version and exit
  --help                               Print help  
  --autowb                             One-shot auto white balance calibration (gray 127) and exit
  --autogain                           One-shot auto gain calibration (gray 127) and exit
  --autocalib                          One-shot auto white balance + gain calibration (gray 127) and exit
  --autosat arg                        One-shot auto saturation calibration using a target color (R,G,B, default) and exit
  --autosat-iters arg (=24)            Auto saturation max iterations
  --autocalib-iters arg (=8)           Auto calibration max iterations
  --autocalib-delay-ms arg (=500)      Delay between auto calibration updates (ms)
  --autocalib-tol-wb arg (=2)          Auto WB convergence tolerance (abs channel delta: |R-G| and |B-G|)
  --autocalib-tol-grey arg (=3)        Auto gain convergence tolerance (abs avg RGB error from 127)
  --autocalib-gain-samples arg (=8)    Auto gain samples (curve-fit)
  --autocalib-gain-range arg (=4)      Auto gain range multiplier (min=gain/range, max=gain*range)
  --autofocus-once                     Run autofocus once at startup, print focus value, then lock manual focus
```

### One-shot image capture (`--imgout`)
Use `--imgout /path/file.png` to capture exactly one frame, save it, and exit.
- If `--transform` is provided, the saved image is the transformed output.
- `--imgout` always saves RGB/BGR images (there is no common still-image container for NV12). If `--nv12` is set, it is ignored.
- Use `--imgout-crophelp /path/file.png` together with `-w` and `-h` to see what sensor area the camera uses at that resolution. This helps determine the correct coordinates for `--crop`. The camera automatically crops to the center when using lower resolutions, and this option shows you exactly what area is being captured, making it easier to adjust the crop position with `--crop x,y,width,height`.

Examples:
```
./hyperion-cam --imgout /tmp/frame.png
./hyperion-cam --transform "0,0:640,0:0,480:640,480" --imgout /tmp/frame.png
./hyperion-cam --cpuonly --transform "..." --imgout /tmp/frame.png
./hyperion-cam --imgout-crophelp /tmp/crop-area.png -w 640 -h 480       # See what 640x480 sensor area is used
./hyperion-cam --imgout-crophelp /tmp/crop-area.png -w 1920 -h 1080 -v  # Verbose shows ScalerCropMaximum coordinates
```

#### Focus note:
For autofocus-capable cameras (e.g. Camera Module 3), the app forces autofocus into manual mode.
Use `--focus` to set the manual lens position.

Use `--autofocus-once` to run autofocus once, print the resulting focus position, and then lock to manual focus. You can reuse then this auto value with the `--focus` option

#### Flip note:
The `--hflip` and `--vflip` options use libcamera's native Transform control to flip the image horizontally or vertically at the hardware level (sensor or ISP). These are zero-overhead operations and work with all pixel formats (RGB, NV12). They can be combined with `--transform` for perspective correction.

#### Rotation note:
Note: The rotation depends on a camera capabilities. Not all cameras support e.g. 90° and 270°
The `--rotate` option provides hardware-accelerated rotation (0°, 90°, 180°, 270°) using libcamera's native Transform. This can be combined with `--hflip` and `--vflip` for complex orientations:

Examples:
```
./hyperion-cam --rotate 90                    # 90° clockwise
./hyperion-cam --rotate 180                   # 180° rotation
./hyperion-cam --rotate 90 --hflip            # 90° + horizontal flip
./hyperion-cam --hflip --vflip                # Both flips (same as --rotate 180)
```

**Note:** Some combinations of rotation and flip produce the same result:
- `--rotate 180` = `--hflip --vflip` (both result in 180° rotation)
- `--rotate 180 --hflip` = `--vflip` alone
- `--rotate 180 --vflip` = `--hflip` alone
- `--rotate 180 --hflip --vflip` = no transform (identity)

The code handles these redundancies automatically and applies the correct libcamera Orientation.

#### Crop note:
The app provides two levels of cropping for different use cases:

**Sensor-level crop (`--crop`):**
- Uses libcamera's ScalerCrop control to crop at the sensor/ISP level
- Coordinates are in **absolute sensor coordinates** (not relative to (0,0))
- Applied before any transforms, rotation, or flip operations on chip level
- Hardware-accelerated, very efficient
- Can be combined with `-w` and `-h` for downscaling: sensor crop happens first (reducing bandwidth), then the output is scaled to requested resolution
- Format: `x,y,width,height` (e.g., `--crop 768,432,1920,1080`)
- Use `-v` to see `ScalerCropMaximum` which shows the valid sensor coordinate range (e.g., `(768, 432)/3072x1728`)
- Examples:
  ```bash
  ./hyperion-cam --crop 768,432,1920,1080                    # Crop 1920x1080 region, output 640x480 (default, downscaled)
  ./hyperion-cam --crop 768,432,1920,1080 -w 1920 -h 1080    # Crop 1920x1080 region, output 1920x1080 (1:1, no scaling)
  ./hyperion-cam --crop 768,432,2000,2000 -w 500 -h 500      # Crop 2000x2000, downscale to 500x500
  ./hyperion-cam -v --crop 768,432,1000,1000 -w 640 -h 640   # Verbose shows ScalerCropMaximum
  ```

**Output-level crop (`--output-crop`):**
- Crops the final output image (only for non-transformed images)
- **Cannot be used with `--transform`** (they are mutually exclusive)
  - If using perspective correction, adjust the 4-point transform coordinates instead
  - The transform already defines your output region via perspective mapping
- Coordinates are relative to the camera output (after rotation/flip if used)
- Software-based crop, adds minimal overhead (simple memcpy)
- Useful for cropping raw camera output to specific region of interest
- Format: `x,y,width,height` (e.g., `--output-crop 10,10,620,460`)
- Example:
  ```bash
  # Crop 640x480 raw output to center 540x380 region
  ./hyperion-cam -w 640 -h 480 --output-crop 50,50,540,380
  ```

**When to use each:**
- Use `--crop` for selecting a specific sensor region before any processing (very effective)
- Use `--output-crop` for cropping final output when NOT using `--transform`
- For perspective correction, use `--transform` alone (defines output region via 4 points)
- `--crop` and `--output-crop` can be combined, but `--output-crop` and `--transform` cannot

#### Shutter speed and FPS note:
The `--shutter` speed modifies the exposition time of the camera. When used alone, it defines also the FPS, frame is released immediatelly after capture (recommended mode).
The `--fps` option specify exact frame rate using libcamera's `FrameDurationLimits` control. When set, the camera will maintain the specified frame rate.

**Important:** FPS also limits the maximum shutter speed. The shutter speed will never be longer than the time available for a single frame at the specified FPS. For example:
- At 30 FPS, the frame duration is ~33.3ms, so shutter speed cannot exceed 33333 µs
- At 60 FPS, the frame duration is ~16.7ms, so shutter speed cannot exceed 16666 µs

But you can use faster shutter speed together with lower FPS, in such case next frame is delayed to maintain the desired FPS.
This also produce lower latency (e.g. 30 FPS + shutter 16666 us (60fps)).

#### Connection note:
The app supports the use of a TCP connection or Unix Domain Socket (available when using [HyperHDR](https://github.com/awawa-dev/HyperHDR) at /tmp/hyperhdr-domain). If `--connection` is a valid IP address (IPv4/IPv6) it uses TCP (`--port` applies). Otherwise it treats the value as a Unix socket path. IPC has advantage of faster data exchange with lower overhead.  
I don't recommend to use the TCP connection over a network as you will probably have lags (due to high packet size and packet framing). Maybe on an ethernet network it would be usable, but over wifi it's unusable. Rather always install Hyperion locally and then send only the LED data to another Hyperion server (if you can't drive the LEDs directly).

#### Auto calibration note:
For one-time setup, the app can estimate white balance gains and/or analogue gain using a neutral gray target.

- Display or place a uniform gray image with RGB value `127,127,127` in front of the camera.
- The app captures frames in RGB mode (NV12 is ignored).
- It measures the average color of an outline box ROI made from the image edges (optionally transformed): top/bottom/left/right strips, each with thickness = 1/8 of image size.
  - Example: for 640x480 it uses 80px strips (top/bottom) and 60px strips (left/right).
- It iteratively adjusts `--wbgain` (R,B) to make R≈G≈B and `--gain` to reach ≈127 brightness, prints the final values and the final measured RGB, then exits.
  - Use `--autocalib-tol-wb` and `--autocalib-tol-grey` to require tighter convergence.

Auto gain method:
- Auto gain uses curve fitting: it samples multiple analogue gain values, fits a curve to the measured ROI brightness, solves for the gain that yields ~127, then confirms on a final frame.

Examples:
```
./hyperion-cam --autowb
./hyperion-cam --autogain
./hyperion-cam --autocalib
./hyperion-cam --transform "0,0:640,0:0,480:640,480" --autocalib
```

#### Auto saturation note:
For one-time setup, the app can estimate camera saturation using a target color.

- Display or place a uniform target image with RGB value `230,50,50` in front of the camera.
  - You can override the target color by passing `R,G,B` to `--autosat`, e.g. `--autosat 50,230,50` for green.
- The app captures frames in RGB mode (NV12 is ignored).
- It measures the average color of an outline box ROI made from the image edges (optionally transformed): top/bottom/left/right strips, each with thickness = 1/8 of image size.
- It searches saturation so the measured ROI color matches the target color (within a small tolerance), then prints the final saturation and final measured RGB, then exits.
- Don't use values near the end because of clipping (like 255, 0, 0)
- For some reason a blue target doesn't work as well, but with red and green targets I get the same result.

Examples:
```
./hyperion-cam --autosat
./hyperion-cam --autosat 50,230,50
./hyperion-cam --transform "0,0:640,0:0,480:640,480" --autosat
./hyperion-cam --transform "0,0:640,0:0,480:640,480" --autosat 50,230,50
```

#### Gamma note:
Not all cameras support gamma correction.

#### Picture transformation note:
The app allows to make Perspective (projective) transformation of the picture in real time using OpenCV. Of course, it adds a little lag, but it's numbers of milliseconds.  
The transformation is described by 4 points in the original picture. The result is cropped and transformed so the 4 points make new corners of image.

### GPU transform (optional)
If built with EGL + OpenGL ES support, GPU transform is used by default whenever `--transform` is provided.
Use `--cpuonly` to force CPU transforms.

### Low latency tuning
This app runs capture/transform/send asynchronously. If the CPU/GPU or HyperHDR connection can't keep up, the internal queues can grow which increases end-to-end latency.

To keep latency bounded (at the cost of dropping frames under load), use:
- `--max-pending-process 1`: don't queue transforms; drop frames while busy
- `--max-pending-send 1` (or `2`): don't queue sends; drop frames while busy

If HyperHDR is restarting or the socket is temporarily down, the sender can otherwise block on connect attempts. To keep the send loop responsive, tune:
- `--connect-timeout-ms 250` (or lower): limits how long a single connect attempt can block
- `--reconnect-interval-ms 250` (or higher): limits how often reconnect attempts are made while disconnected

Examples:
```
# Lowest latency (bounded) when using a transform
./hyperion-cam --transform "..." --max-pending-process 1 --max-pending-send 1

# Lowest latency (bounded) without transform
./hyperion-cam --max-pending-send 1

# When HyperHDR restarts frequently (avoid connect-stalls/spam)
./hyperion-cam --max-pending-send 1 --connect-timeout-ms 100 --reconnect-interval-ms 250
```

### Performance / profiling knobs
These are optional environment variables used mainly for profiling and tuning:
- `HYPERCAM_CPU_PROFILE=1`: print average CPU transform time (RGB/NV12)
- `HYPERCAM_GL_PROFILE=1`: print average GPU stage timings
- `HYPERCAM_GL_PROFILE_SYNC=1`: GPU profiling with extra sync (more accurate, slower)
- `HYPERCAM_NV12_PBO=1`: NV12 readback pipeline using PBO+fences (can change latency/jitter)

GPU paths attempt DMA-BUF import automatically (zero-copy upload when supported) and fall back to CPU upload if not available.

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
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . -j
```

The binary will be at `Release/hyperion-cam`.

## Credits
The camera part is based on [simple-cam](https://git.libcamera.org/libcamera/simple-cam.git) example from libcamera  
The OpenCV part is based on [OpenCV example](https://docs.opencv.org/3.4/da/d6e/tutorial_py_geometric_transformations.html)  
The Flatbuffer part is based on [Hyperion for LG WebOS](https://github.com/webosbrew/hyperion-webos) and [Hyperion-ng](https://github.com/hyperion-project/hyperion.ng).
