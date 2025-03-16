#include "downloader/downloader.h"
#include "audio/converter.h"
#include "transcriber/transcriber.h"
#include <iostream>
#include <string>
#include <fstream>

// Dodaj odpowiednie nagłówki dla GPU
#ifdef WHISPER_USE_CUDA
#include <cuda_runtime.h>
#endif

// Zmień sposób dołączania OpenCL
#ifdef WHISPER_USE_OPENCL
#include <CL/cl.h>
#endif


// Funkcja do zapisu transkrypcji do pliku
void saveTranscriptionToFile(const std::string& text, const std::string& filePath) {
    std::ofstream outFile(filePath);
    if (outFile.is_open()) {
        outFile << text;
        outFile.close();
        std::cout << "Transkrypcja zapisana do: " << filePath << std::endl;
    } else {
        std::cerr << "Nie można zapisać transkrypcji do pliku: " << filePath << std::endl;
    }
}

// Funkcja do sprawdzenia dostępności GPU
bool isGPUAvailable() {
    bool gpuAvailable = false;
    
#ifdef WHISPER_USE_CUDA
    // Sprawdzanie dostępności CUDA
    int deviceCount = 0;
    cudaError_t error = cudaGetDeviceCount(&deviceCount);
    if (error == cudaSuccess && deviceCount > 0) {
        std::cout << "Wykryto " << deviceCount << " urządzeń CUDA" << std::endl;
        gpuAvailable = true;
    }
#endif

#ifdef WHISPER_USE_OPENCL
    // Używamy niskopoziomowego API OpenCL zamiast cl:: namespace
    cl_uint platformCount = 0;
    cl_int result = clGetPlatformIDs(0, nullptr, &platformCount);
    
    if (result == CL_SUCCESS && platformCount > 0) {
        std::cout << "Wykryto " << platformCount << " platform OpenCL" << std::endl;
        gpuAvailable = true;
    }
#endif

    return gpuAvailable;
}

// Funkcja wyświetlająca pasek postępu
void showProgressBar(float progress) {
    const int barWidth = 50;
    std::cout << "[";
    int pos = barWidth * progress;
    for (int i = 0; i < barWidth; ++i) {
        if (i < pos) std::cout << "=";
        else if (i == pos) std::cout << ">";
        else std::cout << " ";
    }
    std::cout << "] " << int(progress * 100.0) << " %\r";
    std::cout.flush();
}

int main(int argc, char* argv[]) {
    try {
        // Utwórz obiekt klasy Downloader
        Downloader downloader;
        
        // Ustaw katalog wyjściowy
        downloader.setOutputDirectory("./downloads");
        
        // Pobranie adresu URL z argumentów wiersza poleceń lub od użytkownika
        std::string url;
        if (argc > 1) {
            url = argv[1];
        } else {
            std::cout << "Podaj URL filmu YouTube: ";
            std::getline(std::cin, url);
        }
        
        // Ustawienie URL i pobranie informacji o filmie
        downloader.setUrl(url);
        
        // Wyświetl informacje o filmie
        std::cout << "Tytuł: " << downloader.getVideoTitle() << std::endl;
        
        // Rozpocznij pobieranie
        std::string audioFilePath = downloader.startDownload();
        std::cout << "Pobieranie zakończone: " << audioFilePath << std::endl;
        
        // Zapytaj czy chcesz transkrybować audio
        std::string choice;
        std::cout << "Czy chcesz dokonać transkrypcji audio do tekstu? (t/n): ";
        std::getline(std::cin, choice);
        
        if (choice == "t" || choice == "T") {
            try {
                // Wybór metody transkrypcji
                bool useGPU = false;
                bool isGPUSupported = isGPUAvailable();
                
                
                if (isGPUSupported) {
                    std::cout << "Wykryto obsługę GPU. Czy chcesz użyć GPU do transkrypcji? (t/n): ";
                    std::string gpuChoice;
                    std::getline(std::cin, gpuChoice);
                    useGPU = (gpuChoice == "t" || gpuChoice == "T");
                } else {
                    std::cout << "Obsługa GPU nie jest dostępna. Transkrypcja będzie wykonana na CPU." << std::endl;
                }
                
                // Wybór rozmiaru modelu
                ModelSize selectedModel = ModelSize::MEDIUM;
                std::cout << "Wybierz rozmiar modelu:" << std::endl;
                std::cout << "1. Tiny (najszybszy, najmniej dokładny)" << std::endl;
                std::cout << "2. Base (szybki, mniej dokładny)" << std::endl;
                std::cout << "3. Small (średnio szybki, średnia dokładność)" << std::endl;
                std::cout << "4. Medium (wolny, dokładny)" << std::endl;
                std::cout << "5. Large (bardzo wolny, najdokładniejszy)" << std::endl;
                std::cout << "Wybór (1-5, domyślnie 4): ";
                
                std::string modelChoice;
                std::getline(std::cin, modelChoice);
                
                if (modelChoice == "1") selectedModel = ModelSize::TINY;
                else if (modelChoice == "2") selectedModel = ModelSize::BASE;
                else if (modelChoice == "3") selectedModel = ModelSize::SMALL;
                else if (modelChoice == "5") selectedModel = ModelSize::LARGE;
                // domyślnie MEDIUM dla wszystkich innych opcji
                
                // Informacja o modelu i urządzeniu
                std::cout << "Wybrano model: ";
                switch (selectedModel) {
                    case ModelSize::TINY: std::cout << "Tiny"; break;
                    case ModelSize::BASE: std::cout << "Base"; break;
                    case ModelSize::SMALL: std::cout << "Small"; break;
                    case ModelSize::MEDIUM: std::cout << "Medium"; break;
                    case ModelSize::LARGE: std::cout << "Large"; break;
                }
                std::cout << " do uruchomienia na " << (useGPU ? "GPU" : "CPU") << std::endl;
                
                // Inicjalizacja modułu transkrypcji
                Transcriber transcriber(selectedModel, useGPU);
                
                // Sprawdź czy model jest dostępny, jeśli nie - pobierz go
                if (!transcriber.isAvailable()) {
                    std::cout << "Model transkrypcji nie jest jeszcze pobrany." << std::endl;
                    if (!transcriber.downloadModelIfNeeded()) {
                        std::cerr << "Nie udało się pobrać modelu. Transkrypcja nie będzie możliwa." << std::endl;
                        return 1;
                    }
                }
                
                // Ustaw callback do śledzenia postępu
                transcriber.setProgressCallback(showProgressBar);
                
                // Wybór języka transkrypcji
                bool forcePolish = false;
                std::cout << "Wymuszić język polski? (automatyczne wykrywanie, jeśli nie) (t/n): ";
                std::string langChoice;
                std::getline(std::cin, langChoice);
                forcePolish = (langChoice == "t" || langChoice == "T");
                
                std::cout << "\nRozpoczynanie transkrypcji... To może zająć kilka minut." << std::endl;
                
                // Wykonaj transkrypcję
                std::string transcription = transcriber.transcribeAudio(audioFilePath, forcePolish);
                
                // Wyświetl fragment transkrypcji
                std::cout << "\nFragment transkrypcji:" << std::endl;
                std::cout << transcription.substr(0, std::min(size_t(200), transcription.length()));
                if (transcription.length() > 200) std::cout << "...";
                std::cout << std::endl;
                
                // Zapisz transkrypcję do pliku
                std::string transcriptionFilePath = audioFilePath.substr(0, audioFilePath.find_last_of('.')) + ".txt";
                saveTranscriptionToFile(transcription, transcriptionFilePath);
                
            } catch (const TranscriberException& e) {
                std::cerr << "Błąd transkrypcji: " << e.what() << std::endl;
            }
        }
        
        return 0;
        
    } catch (const DownloaderException& e) {
        std::cerr << "Błąd pobierania: " << e.what() << std::endl;
        return 1;
    } catch (const std::exception& e) {
        std::cerr << "Wystąpił nieoczekiwany błąd: " << e.what() << std::endl;
        return 1;
    }
}