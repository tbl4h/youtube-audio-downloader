# YouTube Audio Downloader

This project is a simple application that allows users to download audio from YouTube videos by providing the appropriate video link. It utilizes cURL for handling HTTP requests and FFmpeg for audio processing.

## Features

- Download audio from YouTube videos.
- Supports various audio formats through FFmpeg.
- Simple command-line interface for user interaction.

## Requirements

- CMake
- cURL
- FFmpeg

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/youtube-audio-downloader.git
   cd youtube-audio-downloader
   ```

2. Install the required dependencies:
   - For **cURL**: Follow the installation instructions on the [cURL website](https://curl.se/download.html).
   - For **FFmpeg**: Follow the installation instructions on the [FFmpeg website](https://ffmpeg.org/download.html).

3. Build the project using CMake:
   ```
   mkdir build
   cd build
   cmake ..
   make
   ```

## Usage

Run the application from the command line:
```
./youtube-audio-downloader
```

You will be prompted to enter the YouTube video link. After entering the link, the application will start downloading the audio.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.

sudo snap install yt-dlp 
sudo apt install ffmpeg
git clone https://github.com/ggerganov/whisper.cpp.git
sudo apt install -y ocl-icd-opencl-dev ocl-icd-libopencl1 opencl-headers clinfo
sudo apt install -y mesa-opencl-icd
git clone https://github.com/CNugteren/CLBlast.git

# Kompilacja whisper.cpp dla kart Intel Arc

Aby skutecznie wykorzystać kartę graficzną Intel Arc do przyspieszenia transkrypcji, należy poprawnie skompilować bibliotekę whisper.cpp z obsługą OpenCL, unikając problemów z kompatybilnością.

## Wymagania wstępne

- Zainstalowane sterowniki Intel dla OpenCL
- Pakiety deweloperskie OpenCL
- CMake w wersji 3.10 lub nowszej
- Kompilator C++ obsługujący C++17

## Instalacja zależności

```bash
# Instalacja pakietów OpenCL
sudo apt-get update
sudo apt-get install -y ocl-icd-opencl-dev ocl-icd-libopencl1 opencl-headers clinfo

# Instalacja sterowników Intel dla OpenCL
sudo apt-get install -y intel-opencl-icd

## Kompilacja whisper-cpp
# Najpierw wyczyść poprzednie buildy
make clean

# Skonfiguruj z włączonym OpenCL i wyłączonymi kernelami Adreno
cmake -B build -DWHISPER_OPENCL=ON -DGGML_OPENCL_USE_ADRENO_KERNELS=OFF

# Skompiluj
cmake --build build --config Release -j

# Zainstaluj bibliotekę
sudo cmake --install build