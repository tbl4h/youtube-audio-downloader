#include "converter.h"
#include <iostream>
#include <stdexcept>
#include <cstdlib>
#include <array>
#include <memory>
#include <sstream>

ConverterException::ConverterException(const std::string& message) 
    : std::runtime_error(message) {}

Converter::Converter() {
    supportedFormats = {"mp3", "aac", "ogg", "wav", "flac", "m4a"};
    
    if (!checkFFmpegInstallation()) {
        throw ConverterException("FFmpeg is not installed or not in PATH. Please install FFmpeg to use audio conversion.");
    }
}

Converter::~Converter() {
    // Cleanup resources if needed
}

bool Converter::checkFFmpegInstallation() const {
    #ifdef _WIN32
    int result = std::system("where ffmpeg >nul 2>nul");
    #else
    int result = std::system("which ffmpeg >/dev/null 2>&1");
    #endif
    return result == 0;
}

bool Converter::isFormatSupported(const std::string& format) const {
    for (const auto& fmt : supportedFormats) {
        if (format == fmt) {
            return true;
        }
    }
    return false;
}

std::vector<std::string> Converter::getSupportedFormats() const {
    return supportedFormats;
}

std::string Converter::buildFFmpegCommand(const std::string& inputFilePath,
                                        const std::string& outputFilePath,
                                        const std::string& format) const {
    std::stringstream cmd;
    
    cmd << "ffmpeg -i \"" << inputFilePath << "\" -vn ";
    
    // Set codec parameters based on format
    if (format == "mp3") {
        cmd << "-codec:a libmp3lame -q:a 2 ";
    } else if (format == "aac") {
        cmd << "-codec:a aac -b:a 192k ";
    } else if (format == "ogg") {
        cmd << "-codec:a libvorbis -q:a 4 ";
    } else if (format == "flac") {
        cmd << "-codec:a flac -compression_level 8 ";
    } else if (format == "wav") {
        cmd << "-codec:a pcm_s16le ";
    } else if (format == "m4a") {
        cmd << "-codec:a aac -b:a 192k ";
    }
    
    cmd << "-y \"" << outputFilePath << "\" 2>/dev/null";
    return cmd.str();
}

void Converter::convertAudio(const std::string& inputFilePath, 
                           const std::string& outputFilePath, 
                           const std::string& format) {
    if (!isFormatSupported(format)) {
        throw ConverterException("Unsupported audio format: " + format + 
                                ". Supported formats are: " + 
                                getSupportedFormats()[0] + ", " + 
                                getSupportedFormats()[1] + "...");
    }
    
    // Check if input file exists
    FILE* file = fopen(inputFilePath.c_str(), "r");
    if (!file) {
        throw ConverterException("Input file does not exist: " + inputFilePath);
    }
    fclose(file);
    
    std::string command = buildFFmpegCommand(inputFilePath, outputFilePath, format);
    std::cout << "Converting audio to " << format << " format..." << std::endl;
    
    int result = std::system(command.c_str());
    
    if (result != 0) {
        throw ConverterException("Failed to convert audio file. FFmpeg returned error code: " + 
                               std::to_string(result));
    }
    
    std::cout << "Audio successfully converted to: " << outputFilePath << std::endl;
}