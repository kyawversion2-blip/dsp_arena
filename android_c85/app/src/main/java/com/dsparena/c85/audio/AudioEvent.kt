package com.dsparena.c85.audio

import kotlinx.serialization.Serializable

/**
 * Standard 5 acoustic DSP features matching index.html:
 * 1. impact: peak RMS amplitude
 * 2. duration: event length in seconds
 * 3. peak_time: time to peak energy in seconds
 * 4. mel_mean_db: average log Mel energy across event frames (dB)
 * 5. impact_freq_hz: peak interpolated resonance frequency (Hz)
 */
@Serializable
data class AudioFeatures(
    val impact: Float,
    val duration: Float,
    val peak_time: Float,
    val mel_mean_db: Float,
    val impact_freq_hz: Float
) {
    fun toMap(): Map<String, Float> = mapOf(
        "impact" to impact,
        "duration" to duration,
        "peak_time" to peak_time,
        "mel_mean_db" to mel_mean_db,
        "impact_freq_hz" to impact_freq_hz
    )
}

/**
 * Acoustic impact event captured from physical contact.
 */
data class AcousticImpactEvent(
    val eventId: String,
    val timestampMs: Long,
    val sampleRateHz: Int,
    val features: AudioFeatures,
    val rawWindowSamples: FloatArray? = null,
    val isRejected: Boolean = false,
    val rejectionReason: String? = null
)
