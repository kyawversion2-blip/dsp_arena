package com.dsparena.c85.audio

import kotlin.math.log10
import kotlin.math.max
import kotlin.math.min
import kotlin.math.pow

/**
 * Mel-scale filterbank compatible with index.html (melBankL / librosa style).
 * Pre-computes triangular filter weights so that active feature extraction
 * performs zero heap allocations.
 *
 * @param sampleRateHz Audio sample rate in Hz (44100)
 * @param nfft FFT size (e.g. 1024 or 2048)
 * @param numFilters Number of mel bands (default 40)
 * @param fMax Upper frequency limit in Hz (default sampleRateHz / 2)
 */
class MelFilterBank(
    val sampleRateHz: Float = 44100f,
    val nfft: Int = 1024,
    val numFilters: Int = 40,
    fMax: Float = sampleRateHz / 2f
) {
    val nBins: Int = nfft / 2 + 1
    // Flattened weights array: index = filterIndex * nBins + binIndex
    val weights: FloatArray = FloatArray(numFilters * nBins)

    init {
        val safeFmax = min(fMax, sampleRateHz / 2f)

        fun hzToMel(f: Float): Float = 2595f * log10(1f + f / 700f)
        fun melToHz(m: Float): Float = 700f * (10f.pow(m / 2595f) - 1f)

        val mMin = hzToMel(0f)
        val mMax = hzToMel(safeFmax)

        val cHz = FloatArray(numFilters + 2)
        for (b in 0 until (numFilters + 2)) {
            val mel = mMin + b.toFloat() * (mMax - mMin) / (numFilters + 1).toFloat()
            cHz[b] = melToHz(mel)
        }

        for (b in 0 until numFilters) {
            val fl = cHz[b]
            val fc = cHz[b + 1]
            val fh = cHz[b + 2]

            val divL = if ((fc - fl) != 0f) (fc - fl) else 1f
            val divR = if ((fh - fc) != 0f) (fh - fc) else 1f

            val wOffset = b * nBins
            for (i in 0 until nBins) {
                val f = i.toFloat() * sampleRateHz / nfft.toFloat()
                var weight = 0f
                if (f in fl..fc) {
                    weight = (f - fl) / divL
                } else if (f in fc..fh) {
                    weight = (fh - f) / divR
                }
                weights[wOffset + i] = weight
            }
        }
    }

    /**
     * Computes the mean Mel dB energy over multiple STFT frames using preallocated scratch buffers.
     * Guaranteed zero-allocation during execution.
     *
     * @param signal Input audio signal segment
     * @param length Effective length of input audio signal
     * @param hopLength Hop size in samples (e.g. 256 or 512)
     * @param window Precomputed Hann window of size [nfft]
     * @param scratchRe Preallocated FloatArray of size [nfft] for real FFT
     * @param scratchIm Preallocated FloatArray of size [nfft] for imag FFT
     * @param scratchMag Preallocated FloatArray of size [nBins] for power spectrum
     */
    fun computeMeanMelDb(
        signal: FloatArray,
        length: Int,
        hopLength: Int,
        window: FloatArray,
        scratchRe: FloatArray,
        scratchIm: FloatArray,
        scratchMag: FloatArray
    ): Float {
        if (length < nfft) {
            return -80.0f
        }

        val nFrames = max(1, (length - nfft) / hopLength + 1)
        var totalDbSum = 0.0
        val eps = 1e-20f

        for (k in 0 until nFrames) {
            val frameOffset = k * hopLength

            // Windowing frame into scratch buffers
            for (i in 0 until nfft) {
                val s = if (frameOffset + i < length) signal[frameOffset + i] else 0f
                scratchRe[i] = s * window[i]
                scratchIm[i] = 0f
            }

            // In-place FFT
            Radix2Fft.fft(scratchRe, scratchIm)

            // Compute power spectral densities
            for (b in 0 until nBins) {
                scratchMag[b] = scratchRe[b] * scratchRe[b] + scratchIm[b] * scratchIm[b]
            }

            // Filterbank integration and log energy
            var frameAcc = 0.0
            for (b in 0 until numFilters) {
                val wOffset = b * nBins
                var energy = 0f
                for (i in 0 until nBins) {
                    energy += scratchMag[i] * weights[wOffset + i]
                }
                val safeEnergy = max(energy, eps)
                frameAcc += 10.0 * log10(safeEnergy.toDouble())
            }

            totalDbSum += (frameAcc / numFilters.toDouble())
        }

        return (totalDbSum / nFrames.toDouble()).toFloat()
    }
}
