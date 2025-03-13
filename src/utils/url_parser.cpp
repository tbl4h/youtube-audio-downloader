#include "url_parser.h"
#include <string>
#include <regex>

std::string parseUrl(const std::string& url) {
    std::regex urlRegex(R"(^(?:https?:\/\/)?(?:www\.)?(?:youtube\.com\/(?:[^\/\n\s]+\/\S+\/|(?:v|e(?:mbed)?)\/|.*[?&]v=)|youtu\.be\/)([a-zA-Z0-9_-]{11}))");
    std::smatch match;

    if (std::regex_search(url, match, urlRegex) && match.size() > 1) {
        return match.str(1); // Return the video ID
    }
    return ""; // Return an empty string if no match is found
}