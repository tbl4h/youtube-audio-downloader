#include "file_handler.h"
#include <fstream>
#include <iostream>
#include <vector>  // Dodany brakujący nagłówek
#include <cstdint>  // Dla uint8_t

bool saveFile(const std::string& filePath, const std::vector<uint8_t>& audioData) {
    std::ofstream outFile(filePath, std::ios::binary);
    
    if (!outFile.is_open()) {
        std::cerr << "Nie można otworzyć pliku do zapisu: " << filePath << std::endl;
        return false;
    }
    
    outFile.write(reinterpret_cast<const char*>(audioData.data()), audioData.size());
    
    if (outFile.fail()) {
        std::cerr << "Błąd podczas zapisu do pliku: " << filePath << std::endl;
        outFile.close();
        return false;
    }
    
    outFile.close();
    return true;
}

std::vector<uint8_t> readFile(const std::string& filePath) {
    std::ifstream inFile(filePath, std::ios::binary);
    std::vector<uint8_t> fileData;
    
    if (!inFile.is_open()) {
        std::cerr << "Nie można otworzyć pliku do odczytu: " << filePath << std::endl;
        return fileData; // Zwraca pusty wektor
    }
    
    // Uzyskaj rozmiar pliku
    inFile.seekg(0, std::ios::end);
    std::streamsize fileSize = inFile.tellg();
    inFile.seekg(0, std::ios::beg);
    
    // Zaalokuj pamięć i odczytaj zawartość pliku
    fileData.resize(fileSize);
    
    if (fileSize > 0) {
        inFile.read(reinterpret_cast<char*>(fileData.data()), fileSize);
    }
    
    if (inFile.fail() && !inFile.eof()) {
        std::cerr << "Błąd podczas odczytu pliku: " << filePath << std::endl;
        fileData.clear();
    }
    
    inFile.close();
    return fileData;
}

bool fileExists(const std::string& filePath) {
    std::ifstream file(filePath);
    return file.good();
}