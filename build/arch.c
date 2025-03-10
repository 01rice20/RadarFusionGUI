
#if defined(__linux__) || defined(linux) || defined(__linux)
    #if defined(__x86_64) || defined(__x86_64__) || defined(__amd64) || defined(_M_X64)
        #error CMAKE_ARCH linux_x64
    #elif defined(__arm__)
        #error CMAKE_ARCH linux_raspi
    #elif defined(__aarch64__)
        #error CMAKE_ARCH linux_aarch64
    #else
        #error CMAKE_ARCH unknown linux
    #endif
#elif defined(_WIN32) || defined(__MINGW64__) || defined(__CYGWIN__)
    #if defined(__i386) || defined(__i386__) || defined(_M_IX86)
        #error CMAKE_ARCH win32_x86
    #elif defined(__x86_64) || defined(__x86_64__) || defined(__amd64) || defined(_M_X64)
        #error CMAKE_ARCH win32_x64
    #else
        #error CMAKE_ARCH unknown windows
    #endif
#else
    #error CMAKE_ARCH unknown
#endif
