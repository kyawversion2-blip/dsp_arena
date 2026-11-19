package com.dsparena.c85.audio

import kotlin.math.PI
import kotlin.math.cos
import kotlin.math.sin

/**
 * Radix-2 Decimation-in-Time Fast Fourier Transform.
 *
 * Implements in-place complex FFT matching lines 886-907 of index.html.
 * Operates strictly on preallocated FloatArray buffers for zero heap allocation
 * during active audio streaming.
 */
object Radix2Fft {

    /**
     * In-place Radix-2 Cooley-Tukey FFT.
     * @param re Real components array (length must be a power of 2)
     * @param im Imaginary components array (length must match re.size)
     */
    fun fft(re: FloatArray, im: FloatArray) {
        val n = re.size
        if (n < 2) return
        require(n and (n - 1) == 0) { "FFT size must be a power of 2, got $n" }
        require(im.size == n) { "Imaginary array size must match real array size" }

        // Bit-reversal permutation (matches index.html fftRadix2)
        var j = 0
        for (i in 1 until n) {
            var bit = n shr 1
            while (j and bit != 0) {
                j = j xor bit
                bit = bit shr 1
            }
            j = j xor bit
            if (i < j) {
                val tempRe = re[i]
                re[i] = re[j]
                re[j] = tempRe

                val tempIm = im[i]
                im[i] = im[j]
                im[j] = tempIm
            }
        }

        // Cooley-Tukey butterfly computations (matches index.html)
        var len = 2
        while (len <= n) {
            val ang = (-2.0 * PI / len).toFloat()
            val wRe = cos(ang.toDouble()).toFloat()
            val wIm = sin(ang.toDouble()).toFloat()
            val half = len shr 1

            var i = 0
            while (i < n) {
                var curRe = 1.0f
                var curIm = 0.0f
                for (k in 0 until half) {
                    val idxA = i + k
                    val idxB = idxA + half

                    val aRe = re[idxA]
                    val aIm = im[idxA]

                    val bRe = re[idxB] * curRe - im[idxB] * curIm
                    val bIm = re[idxB] * curIm + im[idxB] * curRe

                    re[idxA] = aRe + bRe
                    im[idxA] = aIm + bIm

                    re[idxB] = aRe - bRe
                    im[idxB] = aIm - bIm

                    val nextRe = curRe * wRe - curIm * wIm
                    curIm = curRe * wIm + curIm * wRe
                    curRe = nextRe
                }
                i += len
            }
            len = len shl 1
        }
    }

    /**
     * Generates a standard Hann (Hanning) window:
     * w[i] = 0.5 - 0.5 * cos(2 * PI * i / (n - 1))
     */
    fun createHannWindow(n: Int): FloatArray {
        val window = FloatArray(n)
        val denom = (n - 1).toDouble()
        for (i in 0 until n) {
            window[i] = (0.5 - 0.5 * cos(2.0 * PI * i / denom)).toFloat()
        }
        return window
    }

    /**
     * Interpolates peak frequency around the maximum spectral magnitude bin
     * using parabolic / quadratic peak interpolation matching index.html lines 1344-1350.
     *
     * @param re Real FFT coefficients
     * @param im Imaginary FFT coefficients
     * @param sampleRateHz Audio sample rate in Hz (e.g. 44100)
     * @param fMin Minimum frequency bound in Hz (default 200 Hz)
     * @param fMax Maximum frequency bound in Hz (default 21500 Hz or sr/2)
     */
    fun findInterpolatedPeakFrequency(
        re: FloatArray,
        im: FloatArray,
        sampleRateHz: Float,
        fMin: Float = 200f,
        fMax: Float = 21500f
    ): Float {
        val nfft = re.size
        val half = nfft / 2 + 1
        val safeFmax = fMax.coerceAtMost(sampleRateHz / 2f)

        val b0 = (fMin * nfft / sampleRateHz).toInt().coerceIn(1, half - 2)
        val b1 = (safeFmax * nfft / sampleRateHz).toInt().coerceIn(b0 + 1, half - 1)

        var bestBin = b0
        var bestMagSq = -1f

        for (b in b0..b1) {
            val magSq = re[b] * re[b] + im[b] * im[b]
            if (magSq > bestMagSq) {
                bestMagSq = magSq
                bestBin = b
            }
        }

        if (bestBin <= 0 || bestBin >= half - 1) {
            return bestBin * sampleRateHz / nfft
        }

        val y1 = re[bestBin - 1] * re[bestBin - 1] + im[bestBin - 1] * im[bestBin - 1]
        val y2 = bestMagSq
        val y3 = re[bestBin + 1] * re[bestBin + 1] + im[bestBin + 1] * im[bestBin + 1]

        val denom = y1 - 2f * y2 + y3
        val delta = if (denom != 0f) {
            (0.5f * (y1 - y3) / denom).coerceIn(-0.5f, 0.5f)
        } else {
            0f
        }

        val freqHz = (bestBin + delta) * sampleRateHz / nfft
        // Round to 1 decimal place matching index.html Math.round(... * 10) / 10
        return Math.round(freqHz * 10f) / 10f
    }
}
