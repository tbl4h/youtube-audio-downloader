#include "transcriber.h"
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <filesystem>
#include <curl/curl.h>

// Dołączanie nagłówków dla GPU
#ifdef WHISPER_USE_CUDA
#include <cuda_runtime.h>
#endif

#ifdef WHISPER_USE_OPENCL
#include <CL/cl.h>
#endif

// Dołączanie nagłówków whisper - musi być ostatnie!
#include <whisper.h>

#ifndef WHISPER_USE_OPENCL
#define WHISPER_USE_OPENCL
#endif

namespace fs = std::filesystem;

// USUNIĘTO redefinicję whisper_context_params i whisper_init_from_file_with_params

TranscriberException::TranscriberException(const std::string& message)
    : std::runtime_error(message) {}

// Callback dla curl do zapisywania danych
static size_t writeCallback(void* contents, size_t size, size_t nmemb, std::string* s) {
    size_t newLength = size * nmemb;
    try {
        s->append((char*)contents, newLength);
        return newLength;
    } catch(std::bad_alloc& e) {
        return 0;
    }
}

// Callback dla pobierania plików
static size_t writeDataCallback(void* ptr, size_t size, size_t nmemb, void* stream) {
    size_t written = fwrite(ptr, size, nmemb, (FILE*)stream);
    return written;
}

// Konstruktor
Transcriber::Transcriber(ModelSize modelSize, bool useGPU)
    : modelSize(modelSize), whisperContext(nullptr), useGPU(useGPU) {
    
    // Inicjalizacja callback'a postępu domyślną pustą funkcją
    progressCallback = [](float progress) {};
    
    // Domyślny katalog z modelami - sprawdź najpierw system, potem lokalnie
    std::vector<std::filesystem::path> modelPaths = {
        "/usr/local/share/whisper/models", // ścieżka systemowa
        fs::current_path() / "models",     // lokalna ścieżka
        fs::current_path() / ".." / "models"
    };
    
    for (const auto& path : modelPaths) {
        if (fs::exists(path)) {
            modelsDir = path;
            break;
        }
    }
    
    // Jeśli nie znaleziono katalogu modeli, utwórz lokalny
    if (modelsDir.empty()) {
        modelsDir = fs::current_path() / "models";
        if (!fs::exists(modelsDir)) {
            fs::create_directories(modelsDir);
        }
    }
    
    modelPath = (modelsDir / getModelFilename()).string();
    
    // Pobierz model jeśli nie istnieje
    if (!fs::exists(modelPath)) {
        if (!downloadModel()) {
            throw TranscriberException("Nie udało się pobrać modelu Whisper");
        }
    }

    std::cout << "Inicjalizacja transkrybera z modelem " 
              << getModelFilename() 
              << " na " << (useGPU ? "GPU" : "CPU") << std::endl;
}

// Destruktor
Transcriber::~Transcriber() {
    // Zwolnij zasoby whisper.cpp
    if (whisperContext != nullptr) {
        whisper_free(static_cast<struct whisper_context*>(whisperContext));
        whisperContext = nullptr;
    }
}

void Transcriber::setProgressCallback(ProgressCallback callback) {
    progressCallback = callback;
}

std::string Transcriber::getModelFilename() const {
    switch (modelSize) {
        case ModelSize::TINY:
            return "ggml-tiny.bin";
        case ModelSize::BASE:
            return "ggml-base.bin";
        case ModelSize::SMALL:
            return "ggml-small.bin";
        case ModelSize::MEDIUM:
            return "ggml-medium.bin";
        case ModelSize::LARGE:
            return "ggml-large.bin";
        default:
            return "ggml-medium.bin";
    }
}

bool Transcriber::downloadModel() {
    std::string modelName = getModelFilename();
    std::string modelUrl;
    
    // Określ URL dla każdego modelu
    if (modelName == "ggml-tiny.bin") {
        modelUrl = "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-tiny.bin";
    } else if (modelName == "ggml-base.bin") {
        modelUrl = "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.bin";
    } else if (modelName == "ggml-small.bin") {
        modelUrl = "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-small.bin";
    } else if (modelName == "ggml-medium.bin") {
        modelUrl = "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-medium.bin";
    } else if (modelName == "ggml-large.bin") {
        // Dla modelu large możemy potrzebować innego URL
        modelUrl = "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large.bin";
        
        // Alternatywny URL, jeśli powyższy nie działa
        // modelUrl = "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v1.bin";
    } else {
        std::cerr << "Nieznany model: " << modelName << std::endl;
        return false;
    }
    
    std::cout << "Pobieranie modelu " << modelName << " z " << modelUrl << std::endl;
    
    // Przygotuj katalog docelowy
    if (!fs::exists(modelsDir)) {
        fs::create_directories(modelsDir);
    }
    
    // Inicjalizacja libcurl
    CURL* curl = curl_easy_init();
    if (!curl) {
        std::cerr << "Błąd inicjalizacji CURL" << std::endl;
        return false;
    }
    
    // Otwarcie pliku do zapisu
    FILE* fp = fopen(modelPath.c_str(), "wb");
    if (!fp) {
        std::cerr << "Nie można utworzyć pliku: " << modelPath << std::endl;
        curl_easy_cleanup(curl);
        return false;
    }
    
    // Konfiguracja żądania CURL
    curl_easy_setopt(curl, CURLOPT_URL, modelUrl.c_str());
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, fwrite);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, fp);
    curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);
    
    // Dodaj callback do śledzenia postępu, jeśli jest ustawiony
    if (progressCallback) {
        curl_easy_setopt(curl, CURLOPT_NOPROGRESS, 0L);
        curl_easy_setopt(curl, CURLOPT_XFERINFOFUNCTION, [](void* clientp, curl_off_t dltotal, curl_off_t dlnow, curl_off_t ultotal, curl_off_t ulnow) -> int {
            if (dltotal > 0) {
                Transcriber* transcriber = static_cast<Transcriber*>(clientp);
                float progress = static_cast<float>(dlnow) / static_cast<float>(dltotal);
                transcriber->progressCallback(progress);
            }
            return 0; // Kontynuuj pobieranie
        });
        curl_easy_setopt(curl, CURLOPT_XFERINFODATA, this);
    }
    
    // Wykonanie żądania
    CURLcode res = curl_easy_perform(curl);
    
    // Zamknięcie pliku i zwolnienie zasobów
    fclose(fp);
    curl_easy_cleanup(curl);
    
    // Sprawdzenie wyniku
    if (res != CURLE_OK) {
        std::cerr << "Błąd pobierania modelu: " << curl_easy_strerror(res) << std::endl;
        
        // Usuń częściowo pobrany plik
        if (fs::exists(modelPath)) {
            fs::remove(modelPath);
        }
        
        // Jeśli nie udało się pobrać modelu large, spróbuj alternatywną wersję
        if (modelName == "ggml-large.bin" && modelUrl.find("ggml-large.bin") != std::string::npos) {
            std::cout << "Próbuję pobrać alternatywną wersję modelu large..." << std::endl;
            modelUrl = "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v1.bin";
            
            // Ponowna próba z innym URL
            curl = curl_easy_init();
            if (curl) {
                fp = fopen(modelPath.c_str(), "wb");
                if (fp) {
                    curl_easy_setopt(curl, CURLOPT_URL, modelUrl.c_str());
                    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, fwrite);
                    curl_easy_setopt(curl, CURLOPT_WRITEDATA, fp);
                    curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);
                    
                    res = curl_easy_perform(curl);
                    fclose(fp);
                    curl_easy_cleanup(curl);
                    
                    if (res == CURLE_OK) {
                        std::cout << "Model large pobrany pomyślnie z alternatywnego źródła." << std::endl;
                        return true;
                    } else {
                        std::cerr << "Błąd pobierania modelu z alternatywnego źródła: " << curl_easy_strerror(res) << std::endl;
                        if (fs::exists(modelPath)) {
                            fs::remove(modelPath);
                        }
                    }
                }
            }
        }
        
        return false;
    }
    
    std::cout << "Model " << modelName << " pobrany pomyślnie." << std::endl;
    return true;
}

bool Transcriber::downloadModelIfNeeded() {
    if (!fs::exists(modelPath)) {
        return downloadModel();
    }
    return true;
}

bool Transcriber::isAvailable() const {
    return fs::exists(modelPath);
}

bool Transcriber::isGPUAvailable() {
    // Funkcja ta w rzeczywistości wymagałaby użycia API CUDA lub OpenCL
    // Poniżej uproszczona implementacja
    
    #ifdef WHISPER_USE_CUDA
    // Dla CUDA
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);
    return deviceCount > 0;
    #elif defined(WHISPER_USE_OPENCL)
    // Dla OpenCL
    cl_uint platformCount = 0;
    clGetPlatformIDs(0, nullptr, &platformCount);
    return platformCount > 0;
    #else
    return false;
    #endif
}

void Transcriber::printGPUInfo() {
    #ifdef WHISPER_USE_CUDA
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);
    
    for (int i = 0; i < deviceCount; i++) {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, i);
        
        std::cout << "GPU #" << i << ": " << deviceProp.name << std::endl;
        std::cout << "  Pamięć: " << deviceProp.totalGlobalMem / (1024*1024) << " MB" << std::endl;
        std::cout << "  Rdzenie CUDA: " << deviceProp.multiProcessorCount << std::endl;
    }
    #elif defined(WHISPER_USE_OPENCL)
    // Podobna implementacja dla OpenCL
    #endif
}

// Sprawdzanie dostępności GPU
bool Transcriber::hasGPUSupport() {
#ifdef WHISPER_USE_CUDA
    // Sprawdź czy CUDA jest dostępna
    int deviceCount = 0;
    cudaError_t error = cudaGetDeviceCount(&deviceCount);
    if (error == cudaSuccess && deviceCount > 0) {
        return true;
    }
#endif

#ifdef WHISPER_USE_OPENCL
    // Sprawdź czy OpenCL jest dostępny
    cl_uint platformCount = 0;
    cl_int result = clGetPlatformIDs(0, nullptr, &platformCount);
    if (result == CL_SUCCESS && platformCount > 0) {
        return true;
    }
#endif

    return false;
}

// Przetwarzanie pliku audio
std::vector<float> Transcriber::processAudioFile(const std::string& audioFilePath) {
    // Użyj ffmpeg do konwersji do formatu WAV 16kHz mono
    std::string tempWavPath = audioFilePath + ".temp.wav";
    std::string command = "ffmpeg -y -i \"" + audioFilePath + "\" -ar 16000 -ac 1 -c:a pcm_s16le \"" + tempWavPath + "\" 2>/dev/null";
    
    int result = std::system(command.c_str());
    if (result != 0) {
        throw TranscriberException("Nie udało się przekonwertować pliku audio za pomocą ffmpeg");
    }
    
    // Wczytaj plik WAV jako próbki float
    std::vector<float> pcmData;
    
    FILE* fp = fopen(tempWavPath.c_str(), "rb");
    if (!fp) {
        throw TranscriberException("Nie można otworzyć pliku WAV");
    }
    
    // Pomiń nagłówek WAV (44 bajty)
    fseek(fp, 44, SEEK_SET);
    
    // Odczytaj próbki 16-bit
    std::vector<int16_t> samples;
    int16_t sample;
    while (fread(&sample, sizeof(sample), 1, fp) == 1) {
        samples.push_back(sample);
    }
    
    fclose(fp);
    
    // Usuń plik tymczasowy
    std::remove(tempWavPath.c_str());
    
    // Konwertuj 16-bit PCM na float w zakresie [-1, 1]
    pcmData.resize(samples.size());
    for (size_t i = 0; i < samples.size(); ++i) {
        pcmData[i] = static_cast<float>(samples[i]) / 32768.0f;
    }
    
    return pcmData;
}

// Funkcja transkrybująca plik audio
std::string Transcriber::transcribeAudio(const std::string& audioFilePath, bool forcePolish) {
    if (!fs::exists(audioFilePath)) {
        throw TranscriberException("Plik audio nie istnieje: " + audioFilePath);
    }
    
    std::cout << "Rozpoczynam transkrypcję pliku: " << audioFilePath << std::endl;
    
    // Załaduj model jeśli nie jest jeszcze załadowany
    if (whisperContext == nullptr) {
        std::cout << "Ładowanie modelu Whisper..." << std::endl;
        
        // Wyświetlenie informacji o systemie
        std::cout << "Informacje o systemie whisper: " << whisper_print_system_info() << std::endl;
        
        // Dodaj przed inicjalizacją whisper:
        std::cout << "Opóźnienie inicjalizacji OpenCL aby zainicjować GPU...\n";
        std::this_thread::sleep_for(std::chrono::milliseconds(1000));
        
        // Dodaj te zmienne środowiskowe przed inicjalizacją
        setenv("GGML_OPENCL_SYNCHRONIZE", "1", 1);  // Wymusza synchronizację po każdej operacji
        setenv("GGML_OPENCL_SEQUENTIAL", "1", 1);   // Sekwencyjne wykonywanie operacji

        // Próba inicjalizacji z GPU
        struct whisper_context_params params = whisper_context_default_params();
        
        if (useGPU) {
            std::cout << "Próba inicjalizacji modelu na GPU..." << std::endl;
            params.use_gpu = true;
            params.gpu_device = 0;
            
            // Poniższe ustawienia powinny pomóc z kartą Intel Arc A770 16GB
            setenv("GGML_OPENCL_PLATFORM", "Intel(R) OpenCL Graphics", 1);  // 1 zamiast 0, aby wymusić
            setenv("GGML_OPENCL_DEVICE", "Intel(R) Arc(TM) A770 Graphics", 1);
            
            // Dodaj zmienne środowiskowe do kontroli bufora GPU
            setenv("GGML_OPENCL_MAX_BUFFER_SIZE", "4096", 1);  // Limit bufora w MB (opcjonalnie)
            // Dodaj przed uruchomieniem transkrypcji:
            setenv("GGML_OPENCL_MAX_TENSOR_SIZE", "536870912", 1);  // Limit do 512MB
            // Nie używamy split_size_mb, bo nie ma tego parametru w Twojej wersji whisper.cpp
            
            // Spróbuj wymusić prostszą implementację OpenCL
            setenv("GGML_OPENCL_FORCE_BASIC", "1", 1);
            
            whisperContext = whisper_init_from_file_with_params(modelPath.c_str(), params);
            
            if (whisperContext) {
                std::cout << "Model załadowany pomyślnie na GPU" << std::endl;
            } else {
                std::cout << "Nie udało się załadować modelu na GPU, próba CPU..." << std::endl;
                params.use_gpu = false;
                whisperContext = whisper_init_from_file_with_params(modelPath.c_str(), params);
            }
        } else {
            std::cout << "Inicjalizacja modelu na CPU zgodnie z wyborem użytkownika..." << std::endl;
            params.use_gpu = false;
            whisperContext = whisper_init_from_file_with_params(modelPath.c_str(), params);
        }
        
        if (whisperContext == nullptr) {
            throw TranscriberException("Nie udało się załadować modelu Whisper");
        }
    }
    
    // Wczytaj plik audio zamiast używania AudioUtils
    std::vector<float> audioData = processAudioFile(audioFilePath);
    
    // Przygotuj parametry whisper
    struct whisper_full_params wparams = whisper_full_default_params(WHISPER_SAMPLING_GREEDY);
    
    // Ustaw język polski jeśli wymagane
    if (forcePolish) {
        wparams.language = "pl";
        wparams.translate = false;
    }
    
    // Ustaw inne parametry - kluczowe ustawienia dla stabilności
    wparams.print_progress   = true;
    wparams.print_special    = false;
    wparams.print_realtime   = false;
    wparams.print_timestamps = false;
    
    // Ustaw liczbę wątków na maksymalną dostępną w systemie
    wparams.n_threads = std::thread::hardware_concurrency();
    std::cout << "Używam " << wparams.n_threads << " wątków do transkrypcji" << std::endl;
    
    // Kluczowe ustawienia dla stabilności
    wparams.no_context = true;          // Zapobiega błędom kontekstowym
    wparams.max_tokens = 0;             // Bez limitu tokenów
    wparams.audio_ctx = 512;           // Zmniejsz rozmiar kontekstu audio (domyślnie 1500)
    wparams.strategy = WHISPER_SAMPLING_GREEDY; // Najprostsza strategia
    wparams.temperature = 0.0f;         // Deterministic output
    wparams.n_threads = std::min(4, (int)std::thread::hardware_concurrency());  // Ogranicz liczbę wątków dla stabilności

    // Dodatkowe zmienne środowiskowe dla stabilności
    setenv("GGML_OPENCL_DEQUANT", "0", 0);  // Wyłącz dekwantyzację na GPU

    // Uruchamiamy z debugowaniem
    std::cout << "Transkrypcja w toku...";
    fflush(stdout);
    
    try {
        // Wykonaj transkrypcję z dodatkowym zabezpieczeniem
        if (whisper_full(static_cast<struct whisper_context*>(whisperContext), 
                    wparams, 
                    audioData.data(), 
                    audioData.size()) != 0) {
            throw TranscriberException("Nie udało się wykonać transkrypcji");
        }
        
        // Pobierz wynik transkrypcji
        std::string transcription;
        const int n_segments = whisper_full_n_segments(static_cast<struct whisper_context*>(whisperContext));
        
        for (int i = 0; i < n_segments; ++i) {
            const char* text = whisper_full_get_segment_text(static_cast<struct whisper_context*>(whisperContext), i);
            transcription += text;
            transcription += " ";
        }
        
        std::cout << "\nTranskrypcja zakończona!" << std::endl;
        return transcription;
    } catch (const std::exception& e) {
        std::cerr << "\nWystąpił błąd podczas transkrypcji: " << e.what() << std::endl;
        throw TranscriberException("Błąd transkrypcji: " + std::string(e.what()));
    }
}