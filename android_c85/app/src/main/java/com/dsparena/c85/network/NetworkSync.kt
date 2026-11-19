package com.dsparena.c85.network

import android.util.Log
import com.dsparena.c85.audio.AcousticImpactEvent
import com.dsparena.c85.audio.AudioFeatures
import io.ktor.client.*
import io.ktor.client.engine.cio.*
import io.ktor.client.plugins.websocket.*
import io.ktor.websocket.*
import kotlinx.coroutines.*
import kotlinx.coroutines.channels.Channel
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.serialization.Serializable
import kotlinx.serialization.encodeToString
import kotlinx.serialization.json.Json
import java.net.URI

enum class ConnectionStatus {
    DISCONNECTED,
    CONNECTING,
    CONNECTED,
    RECONNECTING,
    FAILED
}

@Serializable
data class FeatureMessage(
    val type: String = "feature_vector",
    val event_id: String,
    val timestamp_ms: Long,
    val sample_rate_hz: Int,
    val features: AudioFeatures,
    val device_info: String = "Android C85"
)

@Serializable
data class HeartbeatAckMessage(
    val type: String = "heartbeat_ack",
    val timestamp_ms: Long = System.currentTimeMillis()
)

@Serializable
data class AudioChunkMessage(
    val type: String = "audio_chunk",
    val sample_rate_hz: Int,
    val count: Int,
    val samples: List<Float>
)

/**
 * WebSocket client manager for streaming acoustic feature vectors and PCM chunks
 * to the local Mac server (ws://<host>:8125/ws).
 */
class NetworkSync(
    private val scope: CoroutineScope = CoroutineScope(Dispatchers.IO + SupervisorJob())
) {
    companion object {
        private const val TAG = "NetworkSync"
        const val DEFAULT_SERVER = "10.0.2.2:8125"
        const val DEFAULT_PATH = "/ws"
    }

    private val json = Json {
        ignoreUnknownKeys = true
        encodeDefaults = true
    }

    private val client = HttpClient(CIO) {
        install(WebSockets) {
            pingInterval = 15_000L
        }
    }

    private val _connectionStatus = MutableStateFlow(ConnectionStatus.DISCONNECTED)
    val connectionStatus: StateFlow<ConnectionStatus> = _connectionStatus.asStateFlow()

    private val _lastErrorMessage = MutableStateFlow<String?>(null)
    val lastErrorMessage: StateFlow<String?> = _lastErrorMessage.asStateFlow()

    private val sendChannel = Channel<String>(capacity = 128)
    private var connectionJob: Job? = null
    private var isStreamingActive = false

    /**
     * Connects to the local server WebSocket.
     *
     * @param hostEndpoint Server host:port (e.g. "10.0.2.2:8125" or "192.168.1.100:8125")
     */
    fun connect(hostEndpoint: String = DEFAULT_SERVER) {
        disconnect()
        isStreamingActive = true

        val (host, port, path) = parseEndpoint(hostEndpoint)

        connectionJob = scope.launch {
            var retryDelayMs = 1000L
            while (isActive && isStreamingActive) {
                try {
                    _connectionStatus.value = if (retryDelayMs > 1000L) ConnectionStatus.RECONNECTING else ConnectionStatus.CONNECTING
                    Log.i(TAG, "Connecting to WebSocket at ws://$host:$port$path ...")

                    client.webSocket(host = host, port = port, path = path) {
                        _connectionStatus.value = ConnectionStatus.CONNECTED
                        _lastErrorMessage.value = null
                        retryDelayMs = 1000L // Reset backoff on successful connect
                        Log.i(TAG, "Connected to WebSocket ws://$host:$port$path")

                        // Outgoing message sender loop
                        val sendJob = launch {
                            for (msg in sendChannel) {
                                send(Frame.Text(msg))
                            }
                        }

                        // Incoming message receiver loop
                        try {
                            for (frame in incoming) {
                                if (frame is Frame.Text) {
                                    val text = frame.readText()
                                    handleIncomingMessage(text, this)
                                }
                            }
                        } finally {
                            sendJob.cancel()
                        }
                    }
                } catch (e: Exception) {
                    if (!isActive || !isStreamingActive) break
                    val err = "WebSocket connection error: ${e.message}"
                    Log.w(TAG, "$err. Retrying in ${retryDelayMs}ms...")
                    _lastErrorMessage.value = err
                    _connectionStatus.value = ConnectionStatus.RECONNECTING
                    delay(retryDelayMs)
                    retryDelayMs = (retryDelayMs * 2).coerceAtMost(10_000L)
                }
            }
            _connectionStatus.value = ConnectionStatus.DISCONNECTED
        }
    }

    /**
     * Parses hostEndpoint string into host, port, and path.
     */
    private fun parseEndpoint(endpoint: String): Triple<String, Int, String> {
        val clean = endpoint.trim()
        val urlString = if (clean.startsWith("ws://") || clean.startsWith("http://")) {
            clean
        } else {
            "ws://$clean"
        }

        return try {
            val uri = URI(urlString)
            val host = uri.host ?: "10.0.2.2"
            val port = if (uri.port != -1) uri.port else 8125
            val path = if (!uri.path.isNullOrEmpty() && uri.path != "/") uri.path else DEFAULT_PATH
            Triple(host, port, path)
        } catch (e: Exception) {
            Triple("10.0.2.2", 8125, DEFAULT_PATH)
        }
    }

    private suspend fun handleIncomingMessage(text: String, session: WebSocketSession) {
        try {
            if (text.contains("\"type\":\"heartbeat\"") || text.contains("\"heartbeat\"")) {
                val ack = json.encodeToString(HeartbeatAckMessage())
                session.send(Frame.Text(ack))
            }
        } catch (e: Exception) {
            Log.w(TAG, "Error handling incoming frame: ${e.message}")
        }
    }

    /**
     * Sends an extracted acoustic event feature vector to the server.
     */
    fun sendFeatureEvent(event: AcousticImpactEvent) {
        if (_connectionStatus.value != ConnectionStatus.CONNECTED) {
            return
        }

        val msg = FeatureMessage(
            event_id = event.eventId,
            timestamp_ms = event.timestampMs,
            sample_rate_hz = event.sampleRateHz,
            features = event.features
        )

        try {
            val payload = json.encodeToString(msg)
            sendChannel.trySend(payload)
        } catch (e: Exception) {
            Log.e(TAG, "Error encoding feature event: ${e.message}")
        }
    }

    /**
     * Sends a raw PCM chunk for real-time remote oscilloscope / audio recording.
     */
    fun sendAudioChunk(samples: FloatArray, length: Int, sampleRateHz: Int) {
        if (_connectionStatus.value != ConnectionStatus.CONNECTED) {
            return
        }

        val sampleList = ArrayList<Float>(length)
        for (i in 0 until length) {
            sampleList.add(samples[i])
        }

        val msg = AudioChunkMessage(
            sample_rate_hz = sampleRateHz,
            count = length,
            samples = sampleList
        )

        try {
            val payload = json.encodeToString(msg)
            sendChannel.trySend(payload)
        } catch (e: Exception) {
            Log.e(TAG, "Error encoding audio chunk: ${e.message}")
        }
    }

    /**
     * Disconnects from WebSocket server.
     */
    fun disconnect() {
        isStreamingActive = false
        connectionJob?.cancel()
        connectionJob = null
        _connectionStatus.value = ConnectionStatus.DISCONNECTED
    }

    /**
     * Cleans up all Ktor client resources.
     */
    fun release() {
        disconnect()
        client.close()
    }
}
