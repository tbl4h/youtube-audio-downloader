#ifndef WHISPER_WRAPPER_H
#define WHISPER_WRAPPER_H

// Najpierw próbujemy systemowej instalacji
#if __has_include(<whisper.h>)
    #include <whisper.h>
// Jeśli nie ma systemowej, używamy lokalnej
#else
    #include "../third_party/whisper/whisper.h"
#endif

#endif // WHISPER_WRAPPER_H