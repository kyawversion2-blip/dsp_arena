package com.dsparena.c85.audio

import kotlin.math.abs
import kotlin.math.max
import kotlin.math.min
import kotlin.math.pow
import kotlin.math.sqrt

/**
 * Real-time zero-allocation acoustic event feature extractor.
 *
 * Implements the Event-Centric Acoustic Architecture:
 * 1. 1000 Hz 4th-order Butterworth High-Pass Filtering.
 * 2. Energy prominence onset / offset detection (top-db threshold).
 * 3. Strict 1.0s physical rejection ceiling (duration > 1.0s rejected).
 * 4. 1.0s centered window slicing (0.5s pre-trigger + 0.5s post-trigger).
 * 5. Extraction of 5 canonical features matching index.html:
 *    - impact (max RMS)
 *    - duration (seconds)
 *    - peak_time (seconds)
 *    - mel_mean_db (40-band Mel filterbank dB)
 *    - impact_freq_hz (radix-2 FFT peak parabolic interpolation)
 */
class AudioFeatureExtractor(
    val sampleRateHz: Int = 44100,
    val nfft: Int = 1024,
    val hopLength: Int = 256,
    val topDb: Float = 30f,
    val windowSec: Float = 1.0f
) {
    val windowSamples: Int = (windowSec * sampleRateHz).toInt() // 44100 samples
    private val halfWindowSamples: Int = windowSamples / 2 // 22050 samples (0.5s)

    // Preallocated DSP components
    private val hpFilter = HighPass1kHzFilter(sampleRateHz.toFloat())
    private val melFilterBank = MelFilterBank(sampleRateHz.toFloat(), nfft = nfft, numFilters = 40)
    private val hannWindow = Radix2Fft.createHannWindow(nfft)

    // Preallocated internal scratch buffers to maintain strict zero-allocation
    private val windowBuffer = FloatArray(windowSamples)
    private val filteredBuffer = FloatArray(windowSamples)
    private val maxFrames = windowSamples / 512 + 2
    private val frameEnergies = FloatArray(maxFrames)

    private val scratchRe = FloatArray(nfft)
    private val scratchIm = FloatArray(nfft)
    private val scratchMag = FloatArray(melFilterBank.nBins)

    // Preallocated array for trimmed event slice
    private val eventSlice = FloatArray(windowSamples)

    /**
     * Inspects a candidate audio window for acoustic impact.
     * If an impact transient is detected and satisfies the physical duration ceiling (< 1.0s),
     * extracts and returns the 5 standard features.
     * Returns null if rejected (e.g. duration > 1.0s, motor hum, or sub-threshold ambient noise).
     *
     * @param rawSamples Raw input PCM samples (length should be at least [windowSamples])
     * @param offset Offset in rawSamples
     * @param length Number of samples to analyze
     */
    fun extractEvent(
        rawSamples: FloatArray,
        offset: Int = 0,
        length: Int = windowSamples
    ): AudioFeatures? {
        val analysisLen = min(length, windowSamples)
        if (analysisLen < sampleRateHz * 0.05f) { // Require at least 50ms of audio
            return null
        }

        // 1. Copy to internal window buffer
        System.arraycopy(rawSamples, offset, windowBuffer, 0, analysisLen)

        // 2. High-Pass Filter at 1000 Hz (reject low-frequency table rumble / motor hum)
        hpFilter.reset()
        hpFilter.process(windowBuffer, filteredBuffer, 0, analysisLen)

        // 3. Trim / detect active transient envelope (matching index.html trimWav)
        val frameLen = 2048
        val frameHop = 512
        val numFrames = (analysisLen - frameLen) / frameHop + 1
        if (numFrames < 1) return null

        var maxEnergy = 0f
        for (k in 0 until numFrames) {
            val fOffset = k * frameHop
            var acc = 0f
            for (i in 0 until frameLen) {
                val v = filteredBuffer[fOffset + i]
                acc += v * v
            }
            val energy = acc / frameLen.toFloat()
            frameEnergies[k] = energy
            if (energy > maxEnergy) {
                maxEnergy = energy
            }
        }

        // Noise gate check: minimum energy prominence required
        if (maxEnergy < 1e-6f) {
            return null // Ambient quiet floor
        }

        val threshold = maxEnergy * (10f.pow(-topDb / 10f))
        var k0 = -1
        var k1 = -1
        for (k in 0 until numFrames) {
            if (frameEnergies[k] >= threshold) {
                if (k0 < 0) k0 = k
                k1 = k
            }
        }

        if (k0 < 0 || k1 < k0) {
            return null
        }

        val onsetSample = k0 * frameHop
        val offsetSample = min(analysisLen, k1 * frameHop + frameLen)
        val eventLength = offsetSample - onsetSample
        val eventDurationSec = eventLength.toFloat() / sampleRateHz.toFloat()

        // 4. Strict 1.0s Physical Rejection Ceiling (Event-Centric Guardrail)
        // Mechanical contact events finish in < 0.5s; signals > 1.0s are continuous speech or vibration.
        if (eventDurationSec > 1.0f || eventDurationSec < 0.005f) {
            return null // REJECTED
        }

        // Copy event slice
        System.arraycopy(filteredBuffer, onsetSample, eventSlice, 0, eventLength)

        // 5. Compute max RMS (impact feature) matching index.html
        val maxRms = computeMaxRms(eventSlice, eventLength, frameLen = 2048, frameHop = 512)

        // 6. Compute peak amplitude time (peak_time feature)
        var maxAbsAmp = -1f
        var peakSample = 0
        for (i in 0 until eventLength) {
            val absVal = abs(eventSlice[i])
            if (absVal > maxAbsAmp) {
                maxAbsAmp = absVal
                peakSample = i
            }
        }
        val peakTimeSec = peakSample.toFloat() / sampleRateHz.toFloat()

        // 7. Compute Impact Frequency Hz using Radix-2 FFT + Parabolic Peak Interpolation
        // Uses the first nfft samples of the event (or zero-padded if shorter)
        val fftCopyLen = min(nfft, eventLength)
        for (i in 0 until nfft) {
            scratchRe[i] = if (i < fftCopyLen) eventSlice[i] else 0f
            scratchIm[i] = 0f
        }
        Radix2Fft.fft(scratchRe, scratchIm)
        val impactFreqHz = Radix2Fft.findInterpolatedPeakFrequency(
            re = scratchRe,
            im = scratchIm,
            sampleRateHz = sampleRateHz.toFloat(),
            fMin = 200f,
            fMax = 21500f
        )

        // 8. Compute Mean Mel dB using 40-band filterbank
        val melMeanDb = melFilterBank.computeMeanMelDb(
            signal = eventSlice,
            length = eventLength,
            hopLength = hopLength,
            window = hannWindow,
            scratchRe = scratchRe,
            scratchIm = scratchIm,
            scratchMag = scratchMag
        )

        return AudioFeatures(
            impact = maxRms,
            duration = eventDurationSec,
            peak_time = peakTimeSec,
            mel_mean_db = melMeanDb,
            impact_freq_hz = impactFreqHz
        )
    }

    /**
     * Computes the maximum sliding-window RMS matching index.html maxRms calculation.
     */
    private fun computeMaxRms(
        samples: FloatArray,
        length: Int,
        frameLen: Int = 2048,
        frameHop: Int = 512
    ): Float {
        if (length < frameLen) {
            var sum = 0f
            for (i in 0 until length) {
                sum += samples[i] * samples[i]
            }
            return sqrt(sum / max(1, length).toFloat())
        }

        val numFrames = (length - frameLen) / frameHop + 1
        var bestRms = 0f

        for (k in 0 until numFrames) {
            val offset = k * frameHop
            var acc = 0f
            for (i in 0 until frameLen) {
                val v = samples[offset + i]
                acc += v * v
            }
            val rms = sqrt(acc / frameLen.toFloat())
            if (rms > bestRms) {
                bestRms = rms
            }
        }

        return bestRms
    }
}
