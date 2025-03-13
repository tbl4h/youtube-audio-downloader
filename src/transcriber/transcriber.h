#ifndef TRANSCRIBER_H
#define TRANSCRIBER_H

#include <string>
#include <vector>
#include <stdexcept>
#include <functional>
#include <filesystem>

// Definicje enum
enum class ModelSize {
    TINY,
    BASE,
    SMALL,
    MEDIUM,
    LARGE
};

// Typ callback dla raportowania postępu
using ProgressCallback = std::function<void(float)>;

// Klasa wyjątku
class TranscriberException : public std::runtime_error {
public:
    explicit TranscriberException(const std::string& message);
};

// Klasa transkrybera
class Transcriber {
public:
    explicit Transcriber(ModelSize modelSize = ModelSize::MEDIUM, bool useGPU = false);
    ~Transcriber();
    
    // Ustaw callback dla raportowania postępu
    void setProgressCallback(ProgressCallback callback);
    
    // Sprawdź czy transkryber jest gotowy do użycia
    bool isAvailable() const;

    // Sprawdź czy dostępna jest karta GPU
    bool isGPUAvailable();

    // Wyświetl informacje o dostępnej karcie GPU
    void printGPUInfo();
    
    // Pobierz model jeśli nie istnieje
    bool downloadModelIfNeeded();
    
    // Transkrybuj plik audio
    std::string transcribeAudio(const std::string& audioFilePath, bool forcePolish = false);
    
    // Sprawdź czy GPU jest obsługiwane
    bool hasGPUSupport();
    
private:
    ModelSize modelSize;
    void* whisperContext;
    ProgressCallback progressCallback;
    std::filesystem::path modelsDir;
    std::string modelPath;
    bool useGPU;
    
    // Pobierz nazwę pliku modelu
    std::string getModelFilename() const;
    
    // Pobierz model
    bool downloadModel();
    
    // Przetwórz plik audio
    std::vector<float> processAudioFile(const std::string& audioFilePath);
};

#endif // TRANSCRIBER_H