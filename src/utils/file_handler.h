#ifndef FILE_HANDLER_H
#define FILE_HANDLER_H

#include <string>
#include <vector>
#include <cstdint>

// Zapisuje dane binarne do pliku
bool saveFile(const std::string& filePath, const std::vector<uint8_t>& audioData);

// Odczytuje zawartość pliku do wektora bajtów
std::vector<uint8_t> readFile(const std::string& filePath);

// Sprawdza czy plik istnieje
bool fileExists(const std::string& filePath);

#endif // FILE_HANDLER_H