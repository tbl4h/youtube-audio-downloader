#ifndef CONVERTER_H
#define CONVERTER_H

#include <string>
#include <vector>
#include <stdexcept>


class ConverterException : public std::runtime_error {
public:
    explicit ConverterException(const std::string& message);
};

class Converter {
public:
    Converter();
    ~Converter();
    
    void convertAudio(const std::string& inputFilePath, const std::string& outputFilePath, const std::string& format);
    
    // Additional helpful methods
    bool isFormatSupported(const std::string& format) const;
    std::vector<std::string> getSupportedFormats() const;
    
private:
    std::vector<std::string> supportedFormats;
    bool checkFFmpegInstallation() const;
    std::string buildFFmpegCommand(const std::string& inputFilePath, 
                                  const std::string& outputFilePath, 
                                  const std::string& format) const;
};

#endif // CONVERTER_H