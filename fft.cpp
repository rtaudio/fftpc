
#include "fft.h"

#include <vector>
#include <limits>
#include <math.h>
#include <stdexcept>

namespace fftpc {
	std::unordered_map<int, fft::plan> fft::plans;
	std::mutex fft::planMutex;

	fft::plan fft::getPlan(int size, bool ifft, float *in, float *out) {
		if (size < 0 || size > 1e18)
			throw std::range_error("Size must be between 0 and 1e18");
		return  plan{ ifft
			? fftwf_plan_dft_c2r_1d(size, (fftwf_complex*)in, out, FFTW_MEASURE)
			: fftwf_plan_dft_r2c_1d(size, in, (fftwf_complex*)out, FFTW_MEASURE), ifft ? -size : size
		};
	}

	fft::plan fft::getPlan(int size, bool ifft) {
		// fftwf_plan is not re-entrant
		std::lock_guard<std::mutex> lock(planMutex);

		if (size < 0 || size > 1e18)
			throw std::range_error("Size must be between 0 and 1e18");

		auto pi = plans.find(ifft ? -size : size);
		if (pi != plans.end()) {
			return pi->second;
		}
		
		std::vector<float> r(size), c((size+1)*2);
		auto p = getPlan(size, ifft, (ifft ? c : r).data(), (ifft ? r : c).data());
		
		plans[ifft ? -size : size] = p;

		return p;
	}



	std::vector<complex_float> operator*(const std::vector<complex_float>& v1, const std::vector<complex_float>& v2)
	{
		auto n = v1.size();
		if (n != v2.size()) throw std::invalid_argument("Vectors must be same size!");

		std::vector<complex_float> res(n);

		for (size_t i = 0; i < n; i++) {
			res[i].re = v1[i].re * v2[i].re - v1[i].im * v2[i].im;
			res[i].im = v1[i].re * v2[i].im + v1[i].im * v2[i].re;
			//dest[2 * i] =		src1[2 * i] * src2[2 * i]	  - src1[2 * i + 1] * src2[2 * i + 1]; // re
			//dest[2 * i + 1] = src1[2 * i] * src2[2 * i + 1] + src1[2 * i + 1] * src2[2 * i]; // im						
		}

		return res;
	}

	std::vector<complex_float> operator/(const std::vector<complex_float>& v1, const std::vector<complex_float>& v2)
	{
		auto n = v1.size();
		if (n != v2.size()) throw std::invalid_argument("Vectors must be same size!");

		std::vector<complex_float> res(n);

		complex_float tmp;
		float len;

		for (size_t i = 0; i<n; i++) {
			// compute |v2[i]|
			len = v2[i].re * v2[i].re + v2[i].im * v2[i].im;
			res[i].re = (v1[i].re * v2[i].re + v1[i].im * v2[i].im) / len;
			res[i].im = (v1[i].im * v2[i].re - v1[i].re * v2[i].im) / len;

			// Realteil: Re'[k] = [ Re{dest[k]}*Re{div[k]} + Im{dest[k]}*Im{div[k]} ] / len
			// Imaginärteil: Im'[k] = [ Im{dest[k]}*Re{div[k]} - Re{dest[k]}*Im{div[k]} ] / len
		}

		return res;
	}


	std::vector<float> abs(const std::vector<complex_float> &vector)
	{
		auto n = vector.size();
		std::vector<float> res(n);
		for (size_t i = 0; i < n; i++) {
            res[i] = sqrtf(vector[i].im * vector[i].im + vector[i].re * vector[i].re);
		}
		return res;
	}

	float max(const std::vector<float> &vector)
	{
		auto n = vector.size();
		float max = std::numeric_limits<float>::lowest();
		std::vector<float> res(n);
		for (auto v : vector) {
			if (v > max)
				max = v;
		}
		return max;
	}

	float min(const std::vector<float> &vector)
	{
		auto n = vector.size();
		float min = std::numeric_limits<float>::max();
		std::vector<float> res(n);
		for (auto v : vector) {
			if (v < min)
				min = v;
		}
		return min;
	}


}
