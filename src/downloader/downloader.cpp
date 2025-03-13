#include "downloader.h"
#include <iostream>
#include <cstdlib>
#include <regex>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <array>
#include <memory>
#include <sstream>
#include <thread>
#include <chrono>

namespace fs = std::filesystem;

DownloaderException::DownloaderException(const std::string& message)
    : std::runtime_error(message) {}

Downloader::Downloader() : outputDirectory(".") {
    // Check if youtube-dl (or yt-dlp) is installed
    if (!checkYoutubeDlInstallation()) {
        std::cerr << "Warning: youtube-dl/yt-dlp not found. Please install it to download videos." << std::endl;
    }
    
    // Default progress callback (does nothing)
    progressCallback = [](float progress) {};
}

Downloader::~Downloader() {
    // Cleanup resources if needed
}

void Downloader::setUrl(const std::string& url) {
    if (!isValidYoutubeUrl(url)) {
        throw DownloaderException("Invalid YouTube URL: " + url);
    }
    
    videoUrl = url;
    parseVideoId();
    fetchVideoInfo();
}

void Downloader::setOutputDirectory(const std::string& directory) {
    // Check if directory exists, if not create it
    if (!fs::exists(directory)) {
        try {
            fs::create_directories(directory);
        } catch (const std::exception& e) {
            throw DownloaderException("Failed to create output directory: " + std::string(e.what()));
        }
    }
    
    outputDirectory = directory;
}

void Downloader::setProgressCallback(ProgressCallback callback) {
    progressCallback = callback;
}

std::string Downloader::executeCommand(const std::string& command) const {
    std::array<char, 128> buffer;
    std::string result;
    
    #ifdef _WIN32
    std::unique_ptr<FILE, decltype(&_pclose)> pipe(_popen(command.c_str(), "r"), _pclose);
    #else
    std::unique_ptr<FILE, decltype(&pclose)> pipe(popen(command.c_str(), "r"), pclose);
    #endif
    
    if (!pipe) {
        throw DownloaderException("Failed to execute command: " + command);
    }
    
    while (fgets(buffer.data(), buffer.size(), pipe.get()) != nullptr) {
        result += buffer.data();
    }
    
    return result;
}

bool Downloader::checkYoutubeDlInstallation() const {
    // First try yt-dlp (more modern fork of youtube-dl)
    #ifdef _WIN32
    int result = std::system("where yt-dlp >nul 2>nul");
    if (result == 0) return true;
    result = std::system("where youtube-dl >nul 2>nul");
    #else
    int result = std::system("which yt-dlp >/dev/null 2>&1");
    if (result == 0) return true;
    result = std::system("which youtube-dl >/dev/null 2>&1");
    #endif
    
    return result == 0;
}

bool Downloader::isValidYoutubeUrl(const std::string& url) const {
    // Różne możliwe formaty URL YouTube
    std::vector<std::regex> youtubePatterns = {
        // Standardowy format: https://www.youtube.com/watch?v=VIDEO_ID
        std::regex(R"((https?://)?(www\.)?youtube\.com/watch\?.*v=[\w-]+.*)", std::regex::icase),
        
        // Skrócony format: https://youtu.be/VIDEO_ID
        std::regex(R"((https?://)?(www\.)?youtu\.be/[\w-]+.*)", std::regex::icase),
        
        // Format osadzony: https://www.youtube.com/embed/VIDEO_ID
        std::regex(R"((https?://)?(www\.)?youtube\.com/embed/[\w-]+.*)", std::regex::icase),
        
        // Format playlisty: https://www.youtube.com/playlist?list=PLAYLIST_ID
        // std::regex(R"((https?://)?(www\.)?youtube\.com/playlist\?.*list=[\w-]+.*)", std::regex::icase)
    };

    // Sprawdzenie czy URL pasuje do któregokolwiek z wzorców
    for (const auto& pattern : youtubePatterns) {
        if (std::regex_match(url, pattern)) {
            return true;
        }
    }
    
    return false;
}

void Downloader::parseVideoId() {
    // Extract video ID from URL
    std::regex videoIdRegex;
    std::smatch match;
    
    if (videoUrl.find("youtu.be") != std::string::npos) {
        // Short URL format: https://youtu.be/VIDEO_ID
        videoIdRegex = std::regex(R"(youtu\.be\/([a-zA-Z0-9_-]{11}))");
    } else {
        // Standard URL format: https://www.youtube.com/watch?v=VIDEO_ID
        /*
        Rozłóżmy to wyrażenie na części:
        std::regex(...) - to jest wywołanie konstruktora klasy std::regex
        R"(...)" - to jest format "raw string literal" w C++, który pozwala na używanie znaków specjalnych bez dodatkowych znaków ucieczki
        v=([a-zA-Z0-9_-]{11}) - to jest właściwe wyrażenie regularne
        */
        videoIdRegex = std::regex(R"(v=([a-zA-Z0-9_-]{11}))");
    }
    
    if (std::regex_search(videoUrl, match, videoIdRegex) && match.size() > 1) {
        videoId = match[1].str();
    } else {
        throw DownloaderException("Failed to extract video ID from URL: " + videoUrl);
    }
}

void Downloader::fetchVideoInfo() {
    std::string command;
    
    // Use yt-dlp if available, otherwise try youtube-dl
    if (std::system("which yt-dlp >/dev/null 2>&1") == 0) {
        command = "yt-dlp --get-title --no-playlist \"" + videoUrl + "\"";
    } else {
        command = "youtube-dl --get-title --no-playlist \"" + videoUrl + "\"";
    }
    
    try {
        videoTitle = executeCommand(command);
        
        // Remove newlines and trim the title
        videoTitle.erase(std::remove(videoTitle.begin(), videoTitle.end(), '\n'), videoTitle.end());
        videoTitle.erase(std::remove(videoTitle.begin(), videoTitle.end(), '\r'), videoTitle.end());
        
        // Replace invalid filename characters
        std::regex invalidChars(R"([\\/:*?"<>|])");
        videoTitle = std::regex_replace(videoTitle, invalidChars, "_");
    } catch (const std::exception& e) {
        throw DownloaderException("Failed to fetch video title: " + std::string(e.what()));
    }
}

std::string Downloader::getVideoTitle() const {
    return videoTitle;
}

std::string Downloader::getVideoId() const {
    return videoId;
}

std::string Downloader::startDownload() {
    if (videoUrl.empty()) {
        throw DownloaderException("No URL set. Call setUrl() before startDownload().");
    }
    
    std::string outputPath = fs::path(outputDirectory) / (videoTitle + ".%(ext)s");
    
    // Prepare command to download audio only
    std::string command;
    
    // Use yt-dlp if available (better performance), otherwise use youtube-dl
    if (std::system("which yt-dlp >/dev/null 2>&1") == 0) {
        command = "yt-dlp -f bestaudio --extract-audio --audio-format mp3 --audio-quality 0 "
                  "--no-playlist --progress --output \"" + outputPath + "\" \"" + videoUrl + "\"";
    } else {
        command = "youtube-dl -f bestaudio --extract-audio --audio-format mp3 --audio-quality 0 "
                  "--no-playlist --progress --output \"" + outputPath + "\" \"" + videoUrl + "\"";
    }
    
    std::cout << "Downloading audio from: " << videoTitle << std::endl;
    
    // Start a thread to simulate progress updates while downloading
    bool isDownloading = true;
    std::thread progressThread([&]() {
        int dots = 0;
        float progress = 0.0f;
        while (isDownloading && progress < 100.0f) {
            std::cout << "\rDownloading" << std::string(dots, '.').c_str() << std::flush;
            dots = (dots + 1) % 4;
            progress += 1.0f;
            progressCallback(progress);
            std::this_thread::sleep_for(std::chrono::milliseconds(300));
        }
    });
    
    // Execute download command
    int result = std::system(command.c_str());
    isDownloading = false;
    
    if (progressThread.joinable()) {
        progressThread.join();
    }
    
    if (result != 0) {
        throw DownloaderException("Failed to download audio.");
    }
    
    // Return path to downloaded file
    std::string downloadedPath = fs::path(outputDirectory) / (videoTitle + ".mp3");
    std::cout << "\nDownload complete: " << downloadedPath << std::endl;
    
    return downloadedPath;
}