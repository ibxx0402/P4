Dokumentation for camera 
https://www.raspberrypi.com/documentation/computers/camera_software.html#building-rpicam-apps-without-building-libcamera

Dokumentation for denoise i opencv 
https://docs.opencv.org/3.4/d1/d79/group__photo__denoise.html#ga03aa4189fc3e31dafd638d90de335617

Stage baseret på 
https://github.com/raspberrypi/rpicam-apps/blob/main/post_processing_stages/sobel_cv_stage.cpp


installation 

```
sudo apt remove --purge rpicam-apps
```

kan måske optimere opencv performance ved at opdatere til nyere version i følge 

https://qengineering.eu/install-opencv-on-raspberry-pi.html

```
# kan ikke husk om jeg endte med at bruge denne
apt-get install libevent-dev
```

Hved ikke om man blot kan køre det hele på en gang 

```
sudo apt-get install libopencv-dev
sudo apt install -y libcamera-dev libepoxy-dev libjpeg-dev libtiff5-dev libpng-dev
sudo apt install libavcodec-dev libavdevice-dev libavformat-dev libswresample-dev
sudo apt install -y git
sudo apt install -y python3-pip git python3-jinja2
sudo apt install -y libboost-dev
sudo apt install -y libgnutls28-dev openssl libtiff5-dev pybind11-dev
sudo apt install -y qtbase5-dev libqt5core5a libqt5gui5 libqt5widgets5
sudo apt install -y meson cmake
sudo apt install -y python3-yaml python3-ply
sudo apt install -y libglib2.0-dev libgstreamer-plugins-base1.0-dev
git clone https://github.com/raspberrypi/libcamera.git
cd libcamera
meson setup build --buildtype=release -Dpipelines=rpi/vc4,rpi/pisp -Dipas=rpi/vc4,rpi/pisp -Dv4l2=true -Dgstreamer=enabled -Dtest=false -Dlc-compliance=disabled -Dcam=disabled -Dqcam=disabled -Ddocumentation=disabled -Dpycamera=enabled
ninja -C build
sudo ninja -C build install
sudo apt install -y cmake libboost-program-options-dev libdrm-dev libexif-dev
sudo apt install -y meson ninja-build
cd ..
git clone https://github.com/raspberrypi/rpicam-apps.git
cd rpicam-apps
meson setup build -Denable_libav=disabled -Denable_drm=enabled -Denable_egl=disabled -Denable_qt=disabled -Denable_opencv=enabled -Denable_tflite=disabled -Denable_hailo=disabled
meson compile -C build
sudo meson install -C build
sudo ldconfig
```

Tilføjning af denoise stage til camera 

```
cd '/home/comtek450/rpicam-apps/post_processing_stages'
```

tilføj fast_cv_denoise_stage.cpp fra github, derefter tilføj følgende linje til meson.build omkring linje 47 

```
'fast_cv_denoise_stage.cpp',
```

gem filen

```
cd .. 

cd assets 
```

tilføj fast_cv_denoise.json til mappen, herefter gå tilbage 

```
cd ..
meson setup build -Denable_libav=disabled -Denable_drm=enabled -Denable_egl=disabled -Denable_qt=disabled -Denable_opencv=enabled -Denable_tflite=disabled -Denable_hailo=disabled

meson compile -C build

sudo meson install -C build

sudo ldconfig
```

test at det virker, erstat med rigtigt path til json fil.  

```
rpicam-still -o test.jpg --post-process-file='/home/comtek450/fast_cv_denoise.json'
```

man kan også streame over udp, husk at ændre ip 

```
rpicam-vid -t 0 -n --inline -o udp://192.168.0.107:9999 --post-process-file='/home/comtek450/rpicam-apps/assets/fast_cv_denoise.json'

```

for at modtage video stream kan man bruge ffmpeg tjek hvordan man skal installere på egent system 

```
ffplay udp://@:9999 -fflags nobuffer -flags low_delay -framedrop

```

man kan også bruge vlc 

```
rpicam-vid -t 0 -n --codec libav --libav-format mpegts -o udp://<ip-addr>:<port>


vlc udp://@:<port>
```

Hvis man ønsker at ændre på encoding, framerate, size, denoise. 
```
rpicam-vid --level 4 --framerate 15 --width 1280 --height 720 --save-pts timestamp.pts --denoise cdn_off -n -t 0 -n --inline -o udp://192.168.0.107:9999 --post-process-file='/home/comtek450/rpicam-apps/assets/fast_cv_denoise.json'
```