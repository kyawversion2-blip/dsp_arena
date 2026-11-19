package com.dsparena.c85.audio

import android.annotation.SuppressLint
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import android.os.Process
import android.util.Log
import kotlinx.coroutines.*
import kotlinx.coroutines.flow.MutableSharedFlow
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.SharedFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asSharedFlow
import kotlinx.coroutines.flow.asStateFlow
import java.util.UUID
import java.util.concurrent.atomic.AtomicBoolean
import kotlin.math.max
import kotlin.math.sqrt

/**
 * High-performance, zero-allocation real-time Audio Engine for Android C85.
 *
 * Configured strictly for:
 * - 44.1 kHz sample rate
 * - 16-bit PCM Mono
 * - Zero-allocation FloatArray ring buffer during recording loops
 * - Monotonic high-precision timing (System.nanoTime() / perf_counter equivalent)
 * - Real-time acoustic trigger detection and 5-feature extraction
 */
class AudioEngine(
    val sampleRateHz: Int = 44100,
    private val chunkSamples: Int = 1024,
    private val triggerThresholdRms: Float = 0.04f,
    private val triggerHoldOffMs: Long = 250L
) {
    companion object {
        private const val TAG = "AudioEngine"
        const val WINDOW_SECONDS = 1.0f
    }

    private val isRecording = AtomicBoolean(false)
    private var recordThread: Thread? = null
    private var dspThread: Thread? = null

    private var audioRecord: AudioRecord? = null

    // Zero-allocation ring buffer (~2.97s capacity @ 44.1kHz)
    val ringBuffer = AudioRingBuffer(capacityPowerOfTwo = 17)

    // Feature extractor with 1000 Hz HP filter and 1.0s physical rejection ceiling
    private val featureExtractor = AudioFeatureExtractor(
        sampleRateHz = sampleRateHz,
        nfft = 1024,
        hopLength = 256,
        topDb = 30f,
        windowSec = WINDOW_SECONDS
    )

    // Preallocated buffers for audio capture loop
    private val shortBuffer = ShortArray(chunkSamples)
    private val floatBuffer = FloatArray(chunkSamples)

    // Preallocated window buffer for DSP feature extraction (1.0s = 44100 samples)
    private val dspWindowSamples = (WINDOW_SECONDS * sampleRateHz).toInt()
    private val dspWindowBuffer = FloatArray(dspWindowSamples)

    // Reactive state flows for Compose UI
    private val _isRecordingFlow = MutableStateFlow(false)
    val isRecordingFlow: StateFlow<Boolean> = _isRecordingFlow.asStateFlow()

    private val _currentRmsFlow = MutableStateFlow(0f)
    val currentRmsFlow: StateFlow<Float> = _currentRmsFlow.asStateFlow()

    private val _peakRmsFlow = MutableStateFlow(0f)
    val peakRmsFlow: StateFlow<Float> = _peakRmsFlow.asStateFlow()

    private val _eventFlow = MutableSharedFlow<AcousticImpactEvent>(extraBufferCapacity = 64)
    val eventFlow: SharedFlow<AcousticImpactEvent> = _eventFlow.asSharedFlow()

    // Real-time audio chunk listener for network streaming
    var onAudioChunkListener: ((pcmFloatChunk: FloatArray, length: Int) -> Unit)? = null
    var onEventListener: ((event: AcousticImpactEvent) -> Unit)? = null

    // High-precision trigger timing tracking (System.nanoTime() monotonic clock)
    @Volatile
    private var lastTriggerNanoTime: Long = 0L

    @Volatile
    private var pendingDspTriggerSampleIndex: Long = -1L

    /**
     * Starts audio recording and DSP feature extraction threads.
     */
    @SuppressLint("MissingPermission")
    @Synchronized
    fun start(): Boolean {
        if (isRecording.get()) {
            Log.w(TAG, "AudioEngine already running.")
            return true
        }

        val channelConfig = AudioFormat.CHANNEL_IN_MONO
        val audioFormat = AudioFormat.ENCODING_PCM_16BIT

        val minBufSize = AudioRecord.getMinBufferSize(sampleRateHz, channelConfig, audioFormat)
        if (minBufSize <= 0) {
            Log.e(TAG, "Invalid AudioRecord minBufferSize: $minBufSize")
            return false
        }

        val bufferSize = max(minBufSize * 2, chunkSamples * 2 * 2)

        try {
            audioRecord = AudioRecord(
                MediaRecorder.AudioSource.MIC,
                sampleRateHz,
                channelConfig,
                audioFormat,
                bufferSize
            )

            if (audioRecord?.state != AudioRecord.STATE_INITIALIZED) {
                Log.e(TAG, "AudioRecord failed to initialize.")
                audioRecord?.release()
                audioRecord = null
                return false
            }

            audioRecord?.startRecording()
            isRecording.set(true)
            _isRecordingFlow.value = true

            ringBuffer.clear()
            lastTriggerNanoTime = 0L
            pendingDspTriggerSampleIndex = -1L

            // 1. Launch high-priority audio record capture thread
            recordThread = Thread(::captureLoop, "AudioRecordCaptureThread").apply {
                priority = Thread.MAX_PRIORITY
                start()
            }

            // 2. Launch synchronized DSP processing thread
            dspThread = Thread(::dspProcessingLoop, "AudioDspWorkerThread").apply {
                priority = Thread.NORM_PRIORITY + 1
                start()
            }

            Log.i(TAG, "AudioEngine successfully started at $sampleRateHz Hz mono.")
            return true
        } catch (e: Exception) {
            Log.e(TAG, "Error starting AudioEngine: ${e.message}", e)
            stop()
            return false
        }
    }

    /**
     * Dedicated real-time audio capture loop.
     * ZERO heap allocation inside the loop for rock-solid deterministic timing.
     */
    private fun captureLoop() {
        Process.setThreadPriority(Process.THREAD_PRIORITY_URGENT_AUDIO)

        var runningPeakRms = 0f
        var decayCounter = 0

        while (isRecording.get()) {
            val record = audioRecord ?: break
            val readCount = record.read(shortBuffer, 0, shortBuffer.size)

            if (readCount <= 0) {
                continue
            }

            // High-precision timing sync (monotonic time.perf_counter equivalent)
            val captureNanoTime = System.nanoTime()

            // Zero-allocation conversion from 16-bit PCM integer to normalized float [-1.0f, 1.0f]
            var sumSquare = 0f
            for (i in 0 until readCount) {
                val sample = shortBuffer[i] / 32768.0f
                floatBuffer[i] = sample
                sumSquare += sample * sample
            }

            // Write into ring buffer
            val writeIndexBefore = ringBuffer.totalSamplesWritten
            ringBuffer.write(floatBuffer, 0, readCount)

            // Calculate instantaneous RMS for UI visualizer
            val currentRms = sqrt(sumSquare / readCount.toFloat())
            if (currentRms > runningPeakRms) {
                runningPeakRms = currentRms
                decayCounter = 0
            } else {
                if (++decayCounter > 10) {
                    runningPeakRms *= 0.95f
                }
            }

            _currentRmsFlow.value = currentRms
            _peakRmsFlow.value = runningPeakRms

            // Stream raw float chunk to network listener if registered
            onAudioChunkListener?.invoke(floatBuffer, readCount)

            // Trigger detection with monotonic hold-off guardrail
            if (currentRms >= triggerThresholdRms) {
                val elapsedMsSinceLastTrigger = (captureNanoTime - lastTriggerNanoTime) / 1_000_000L
                if (elapsedMsSinceLastTrigger >= triggerHoldOffMs) {
                    lastTriggerNanoTime = captureNanoTime
                    // Queue trigger for post-transient window slicing (0.5s after transient midpoint)
                    pendingDspTriggerSampleIndex = writeIndexBefore + readCount / 2
                }
            }
        }
    }

    /**
     * Dedicated DSP worker loop for extracting features from triggered acoustic events.
     * Evaluates physical contact events without stalling the audio capture thread.
     */
    private fun dspProcessingLoop() {
        Process.setThreadPriority(Process.THREAD_PRIORITY_DEFAULT)

        val halfWindow = dspWindowSamples / 2
        while (isRecording.get()) {
            val triggerSample = pendingDspTriggerSampleIndex
            if (triggerSample >= 0L) {
                // Ensure sufficient future context has arrived in the ring buffer (at least 0.5s post-trigger)
                val requiredSamples = triggerSample + halfWindow
                val totalWritten = ringBuffer.totalSamplesWritten

                if (totalWritten >= requiredSamples) {
                    pendingDspTriggerSampleIndex = -1L

                    // Extract exact 1.0s window centered on the trigger point
                    val windowStartSample = max(0L, triggerSample - halfWindow)
                    ringBuffer.copyWindow(windowStartSample, dspWindowBuffer, 0, dspWindowSamples)

                    val eventTimestamp = System.currentTimeMillis()

                    // Extract the 5 canonical acoustic features matching index.html
                    val features = featureExtractor.extractEvent(
                        rawSamples = dspWindowBuffer,
                        offset = 0,
                        length = dspWindowSamples
                    )

                    if (features != null) {
                        val event = AcousticImpactEvent(
                            eventId = UUID.randomUUID().toString().take(8),
                            timestampMs = eventTimestamp,
                            sampleRateHz = sampleRateHz,
                            features = features,
                            rawWindowSamples = dspWindowBuffer.copyOf()
                        )

                        _eventFlow.tryEmit(event)
                        onEventListener?.invoke(event)
                    } else {
                        Log.d(TAG, "Event at sample $triggerSample rejected (duration > 1.0s or ambient noise)")
                    }
                } else {
                    // Context not yet fully buffered, sleep briefly
                    try {
                        Thread.sleep(15)
                    } catch (e: InterruptedException) {
                        break
                    }
                }
            } else {
                try {
                    Thread.sleep(20)
                } catch (e: InterruptedException) {
                    break
                }
            }
        }
    }

    /**
     * Stops audio recording and releases hardware resources.
     */
    @Synchronized
    fun stop() {
        if (!isRecording.getAndSet(false)) {
            return
        }
        _isRecordingFlow.value = false

        try {
            audioRecord?.stop()
        } catch (e: Exception) {
            Log.e(TAG, "Error stopping AudioRecord: ${e.message}")
        }

        try {
            recordThread?.interrupt()
            recordThread?.join(500)
            recordThread = null
        } catch (e: Exception) {
            Log.w(TAG, "Interrupted joining record thread: ${e.message}")
        }

        try {
            dspThread?.interrupt()
            dspThread?.join(500)
            dspThread = null
        } catch (e: Exception) {
            Log.w(TAG, "Interrupted joining dsp thread: ${e.message}")
        }

        try {
            audioRecord?.release()
            audioRecord = null
        } catch (e: Exception) {
            Log.e(TAG, "Error releasing AudioRecord: ${e.message}")
        }

        Log.i(TAG, "AudioEngine stopped.")
    }
}
