cmake_minimum_required(VERSION 3.10)

project(YoutubeAudioDownloader)

set(CMAKE_CXX_STANDARD 17)

# Dodanie opcji kompilacji
option(BUILD_SHARED_LIBS "Build using shared libraries" ON)
option(USE_CUDA "Enable CUDA support" OFF)
option(USE_OPENCL "Enable OpenCL support" ON)

set(EXTRA_LIBS "")

# Szukanie wymaganych pakietów
find_package(CURL REQUIRED)
find_package(Threads REQUIRED)

# Niestandardowe szukanie FFmpeg
find_path(AVCODEC_INCLUDE_DIR libavcodec/avcodec.h)
find_library(AVCODEC_LIBRARY avcodec)
find_path(AVFORMAT_INCLUDE_DIR libavformat/avformat.h)
find_library(AVFORMAT_LIBRARY avformat)
find_path(AVUTIL_INCLUDE_DIR libavutil/avutil.h)
find_library(AVUTIL_LIBRARY avutil)
find_path(SWSCALE_INCLUDE_DIR libswscale/swscale.h)
find_library(SWSCALE_LIBRARY swscale)

# Szukanie biblioteki Whisper zainstalowanej globalnie
find_path(WHISPER_INCLUDE_DIR whisper.h)
find_library(WHISPER_LIBRARY NAMES whisper libwhisper HINTS /usr/local/lib/whisper)

# Sprawdzenie czy FFmpeg został znaleziony
if(NOT AVCODEC_INCLUDE_DIR OR NOT AVCODEC_LIBRARY OR
   NOT AVFORMAT_INCLUDE_DIR OR NOT AVFORMAT_LIBRARY OR
   NOT AVUTIL_INCLUDE_DIR OR NOT AVUTIL_LIBRARY OR
   NOT SWSCALE_INCLUDE_DIR OR NOT SWSCALE_LIBRARY)
  message(FATAL_ERROR "FFmpeg not found. Please install FFmpeg development packages.")
endif()

# Sprawdzenie czy Whisper został znaleziony
if(NOT WHISPER_INCLUDE_DIR OR NOT WHISPER_LIBRARY)
  message(FATAL_ERROR "Whisper library not found. Please install whisper.cpp globally.")
endif()

# Szukanie CUDA jeśli włączone
if(USE_CUDA)
    find_package(CUDA)
    if(CUDA_FOUND)
        include_directories(${CUDA_INCLUDE_DIRS})
        add_definitions(-DWHISPER_USE_CUDA)
        list(APPEND EXTRA_LIBS ${CUDA_LIBRARIES})
        message(STATUS "CUDA support enabled")
    else()
        message(WARNING "CUDA not found, disabling GPU acceleration")
        set(USE_CUDA OFF)
    endif()
endif()

# Szukanie OpenCL jeśli włączone
if(USE_OPENCL)
    find_package(OpenCL)
    if(OpenCL_FOUND)
        # Szukanie CLBlast
        find_library(CLBLAST_LIBRARY clblast)
        if(CLBLAST_LIBRARY)
            include_directories(${OpenCL_INCLUDE_DIRS})
            add_definitions(-DWHISPER_USE_OPENCL)
            list(APPEND EXTRA_LIBS ${OpenCL_LIBRARIES} ${CLBLAST_LIBRARY})  # Poprawiona nazwa zmiennej
            message(STATUS "OpenCL support enabled with CLBlast")
        else()
            message(WARNING "CLBlast not found, disabling OpenCL acceleration")
            set(USE_OPENCL OFF)
        endif()
    else()
        message(WARNING "OpenCL not found, disabling GPU acceleration")
        set(USE_OPENCL OFF)
    endif()
endif()

# Konfiguracja katalogów z plikami nagłówkowymi
include_directories(
    ${CURL_INCLUDE_DIRS}
    ${AVCODEC_INCLUDE_DIR}
    ${AVFORMAT_INCLUDE_DIR}
    ${AVUTIL_INCLUDE_DIR}
    ${SWSCALE_INCLUDE_DIR}
    ${WHISPER_INCLUDE_DIR}
    ${CMAKE_SOURCE_DIR}/include
)

# Zbieranie plików źródłowych projektu
set(SOURCES
    src/main.cpp
    src/downloader/downloader.cpp
    src/utils/file_handler.cpp
    src/audio/converter.cpp
    src/transcriber/transcriber.cpp
    # Usuwamy własne implementacje, ponieważ używamy systemowej biblioteki
    # src/third_party/whisper/whisper.cpp
    # src/third_party/whisper/ggml.cpp
    # src/third_party/whisper/ggml-cpu.cpp
)

# Dodanie pliku wykonywalnego
add_executable(YoutubeAudioDownloader ${SOURCES})

# Linkowanie bibliotek
target_link_libraries(YoutubeAudioDownloader
    ${CURL_LIBRARIES}
    ${AVCODEC_LIBRARY}
    ${AVFORMAT_LIBRARY}
    ${AVUTIL_LIBRARY}
    ${SWSCALE_LIBRARY}
    ${WHISPER_LIBRARY}
    ${CMAKE_THREAD_LIBS_INIT}
    /usr/local/lib/libwhisper.so.1.7.4
    ${EXTRA_LIBS}
)

# Opcje kompilatora
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    target_compile_options(YoutubeAudioDownloader PRIVATE -Wall -Wextra)
endif()

# Instalacja
install(TARGETS YoutubeAudioDownloader DESTINATION bin)

message(STATUS "CURL include dirs: ${CURL_INCLUDE_DIRS}")
message(STATUS "Whisper include dir: ${WHISPER_INCLUDE_DIR}")
message(STATUS "Whisper library: ${WHISPER_LIBRARY}")