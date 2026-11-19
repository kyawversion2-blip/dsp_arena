package com.dsparena.c85.audio

import kotlin.math.PI
import kotlin.math.cos
import kotlin.math.sin

/**
 * 4th-order Butterworth High-Pass Filter implemented as a cascade of two 2nd-order
 * Direct Form II Transposed biquad sections.
 *
 * Configured for 1000 Hz cutoff at 44.1 kHz sample rate matching the Event-Centric
 * acoustic architecture guardrail (rejecting table rumble and motor hum).
 *
 * Implements strict zero-allocation in-place processing for real-time audio streams.
 */
class HighPass1kHzFilter(sampleRateHz: Float = 44100f) {

    private class BiquadSection(
        val b0: Float,
        val b1: Float,
        val b2: Float,
        val a1: Float,
        val a2: Float
    ) {
        var z1: Float = 0f
        var z2: Float = 0f

        fun reset() {
            z1 = 0f
            z2 = 0f
        }

        fun processSample(input: Float): Float {
            // Direct Form II Transposed
            val out = b0 * input + z1
            z1 = b1 * input - a1 * out + z2
            z2 = b2 * input - a2 * out
            return out
        }
    }

    private val section1: BiquadSection
    private val section2: BiquadSection

    init {
        val fc = 1000f
        val omega = (2.0 * PI * fc / sampleRateHz).toFloat()
        val cosOmega = cos(omega.toDouble()).toFloat()
        val sinOmega = sin(omega.toDouble()).toFloat()

        // 4th order Butterworth Q values for 2 cascaded biquads
        val q1 = 0.54119610f
        val q2 = 1.30656296f

        section1 = createHighPassSection(cosOmega, sinOmega, q1)
        section2 = createHighPassSection(cosOmega, sinOmega, q2)
    }

    private fun createHighPassSection(cosOmega: Float, sinOmega: Float, q: Float): BiquadSection {
        val alpha = sinOmega / (2f * q)
        val a0 = 1f + alpha
        val b0 = ((1f + cosOmega) / 2f) / a0
        val b1 = (-(1f + cosOmega)) / a0
        val b2 = ((1f + cosOmega) / 2f) / a0
        val a1 = (-2f * cosOmega) / a0
        val a2 = (1f - alpha) / a0
        return BiquadSection(b0, b1, b2, a1, a2)
    }

    /**
     * Resets internal delay states.
     */
    fun reset() {
        section1.reset()
        section2.reset()
    }

    /**
     * Filters input sample buffer in-place with zero memory allocation.
     */
    fun processInPlace(samples: FloatArray, offset: Int = 0, length: Int = samples.size) {
        val end = offset + length
        for (i in offset until end) {
            val s = samples[i]
            val s1 = section1.processSample(s)
            samples[i] = section2.processSample(s1)
        }
    }

    /**
     * Filters from source array into destination array with zero memory allocation.
     */
    fun process(src: FloatArray, dst: FloatArray, offset: Int = 0, length: Int = src.size) {
        val end = offset + length
        for (i in offset until end) {
            val s = src[i]
            val s1 = section1.processSample(s)
            dst[i] = section2.processSample(s1)
        }
    }
}
