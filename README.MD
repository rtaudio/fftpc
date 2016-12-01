# fftpc

Thread-safe FFT/IFFTP wrapper for FFTW and FFTS. fftpc includes vector arithmetic with complex numbers, using the STL container std::vector.
It can be build with a safe flag so any transformation call makes sure that the containers have enough space.

# Transformations

```
using namespace fftpc;
auto fft = fft::getFFT(N);
auto ifft = fft::getIFFT(N);

std::vector<float> timeData(N);
std::vector<complex_float> freqData(N+1);

fft(timeData, &freqData);
// do stuff with freqData
ifft(freqData, &timeData);
```

Return vector:
```
std::vector<complex_float> freqData = fft(timeData);
```

Convenient autoCorr:
```
auto autoCorr = ifft(fft(timeData) * fft(timeData));
```