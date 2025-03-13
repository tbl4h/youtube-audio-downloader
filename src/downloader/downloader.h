#ifndef DOWNLOADER_H
#define DOWNLOADER_H

#include <string>
#include <vector>
#include <stdexcept>
#include <functional>

class DownloaderException : public std::runtime_error {
public:
    explicit DownloaderException(const std::string& message);
};

class Downloader {
public:
    // Callback function type for progress updates
    using ProgressCallback = std::function<void(float)>;
    
    Downloader();
    ~Downloader();
    
    void setUrl(const std::string& url);
    void setOutputDirectory(const std::string& directory);
    void setProgressCallback(ProgressCallback callback);
    
    // Start downloading and returns the path to the downloaded file
    std::string startDownload();
    
    // Validate if the URL is a valid YouTube URL
    bool isValidYoutubeUrl(const std::string& url) const;
    
    // Get video info from URL
    std::string getVideoTitle() const;
    std::string getVideoId() const;
    
    // Check if youtube-dl is installed
    bool checkYoutubeDlInstallation() const;

private:
    std::string videoUrl;
    std::string outputDirectory;
    std::string videoTitle;
    std::string videoId;
    ProgressCallback progressCallback;
    
    // Parse video ID from URL
    void parseVideoId();
    
    // Extract video information
    void fetchVideoInfo();
    
    // Execute command and get output
    std::string executeCommand(const std::string& command) const;
};

#endif // DOWNLOADER_H