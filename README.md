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
