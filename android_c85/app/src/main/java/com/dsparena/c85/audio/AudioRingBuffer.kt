package com.dsparena.c85.audio

import java.util.concurrent.atomic.AtomicLong

/**
 * Thread-safe, lock-free, zero-allocation ring buffer for single-producer,
 * single-consumer real-time audio sample streaming.
 *
 * Buffer capacity is a power of 2 for fast bitmask indexing without modulo overhead.
 */
class AudioRingBuffer(capacityPowerOfTwo: Int = 17) { // 2^17 = 131,072 samples (~2.97 seconds @ 44.1kHz)
    val capacity: Int = 1 shl capacityPowerOfTwo
    private val mask: Int = capacity - 1

    private val buffer: FloatArray = FloatArray(capacity)
    private val writePos: AtomicLong = AtomicLong(0L)

    /**
     * Total number of samples written to this ring buffer since initialization.
     */
    val totalSamplesWritten: Long
        get() = writePos.get()

    /**
     * Writes incoming PCM float samples into the ring buffer.
     * Guaranteed zero-allocation.
     */
    fun write(src: FloatArray, offset: Int = 0, count: Int = src.size) {
        var currentWrite = writePos.get()
        for (i in 0 until count) {
            val bufIndex = (currentWrite + i).toInt() and mask
            buffer[bufIndex] = src[offset + i]
        }
        writePos.addAndGet(count.toLong())
    }

    /**
     * Reads the most recent [count] samples into [dst] starting at [dstOffset].
     * Zero-allocation.
     *
     * @return Number of samples actually copied.
     */
    fun readLatest(dst: FloatArray, dstOffset: Int = 0, count: Int = dst.size): Int {
        val currentWrite = writePos.get()
        val available = currentWrite.coerceAtMost(capacity.toLong()).toInt()
        val toCopy = count.coerceAtMost(available)

        val startRead = currentWrite - toCopy
        for (i in 0 until toCopy) {
            val bufIndex = (startRead + i).toInt() and mask
            dst[dstOffset + i] = buffer[bufIndex]
        }
        return toCopy
    }

    /**
     * Extracts an exact audio window specified by absolute sample index [startSampleIndex]
     * into [dst] buffer.
     *
     * @param startSampleIndex Monotonic sample index
     * @param dst Destination buffer
     * @param dstOffset Destination buffer offset
     * @param count Number of samples to read
     */
    fun copyWindow(startSampleIndex: Long, dst: FloatArray, dstOffset: Int = 0, count: Int = dst.size) {
        for (i in 0 until count) {
            val bufIndex = (startSampleIndex + i).toInt() and mask
            dst[dstOffset + i] = buffer[bufIndex]
        }
    }

    /**
     * Resets the buffer cursor.
     */
    fun clear() {
        writePos.set(0L)
    }
}
