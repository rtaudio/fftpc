cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
set (CMAKE_CXX_STANDARD 11)


add_library (fft fft.cpp)

find_library(FFTW3F_LIB NAMES fftw3f libfftw3f libfftw3f.so.3 PATHS /usr/lib /usr/local/lib /usr/lib/x86_64-linux-gnu )

target_link_libraries (fft ${FFTW3F_LIB})
target_include_directories (autil PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} ../)

#debug
add_definitions("-g -ggdb")
