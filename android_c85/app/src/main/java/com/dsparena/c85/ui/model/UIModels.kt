package com.dsparena.c85.ui.model

enum class GroupLabel(val id: String, val colorHex: Long) {
    A("A", 0xFF00E5FF), // Cyan
    B("B", 0xFFFF9100), // Amber
    C("C", 0xFF76FF03), // Lime
    D("D", 0xFFE040FB)  // Magenta
}

enum class FeatureType(val key: String, val displayName: String, val unit: String) {
    IMPACT("impact", "Impact Peak", "mag"),
    DURATION("duration", "Duration", "ms"),
    PEAK_TIME("peak_time", "Peak Time", "ms"),
    MEL_MEAN_DB("mel_mean_db", "Mel Mean dB", "dB"),
    IMPACT_FREQ_HZ("impact_freq_hz", "Impact Freq", "Hz")
}

data class DotPoint(
    val id: String,
    val group: GroupLabel,
    val x: Float,
    val y: Float,
    val features: Map<String, Float> = emptyMap(),
    val timestampMs: Long = System.currentTimeMillis()
)

enum class VerdictSource(val label: String, val badge: String) {
    WAV_REAL("WAV-real", "🎙 WAV-real"),
    CSV_SIMULATED("CSV-simulated", "📋 CSV-simulated")
}

data class HonestyMetadata(
    val countsPerGroup: Map<GroupLabel, Int>,
    val source: VerdictSource = VerdictSource.WAV_REAL,
    val sessionRange: String = "249-268",
    val isInferredSession: Boolean = true,
    val isCrossSession: Boolean = false,
    val sampleRateHz: Int = 44100,
    val fmaxHz: Int = 8000,
    val topDb: Float = 30f,
    val windowSec: Float = 0.3f,
    val nFft: Int = 1024,
    val hopLength: Int = 256,
    val honestCeilingPercent: Float = 86f
) {
    val totalCount: Int get() = countsPerGroup.values.sum()
    val hasLowNWarning: Boolean get() = countsPerGroup.values.any { it < 5 }
    val dspParamSummary: String
        get() = "sr=$sampleRateHz, fmax=${fmaxHz}Hz, top_db=${topDb.toInt()}, win=${windowSec}s, n_fft=$nFft, hop=$hopLength"
}

enum class MergeVerdictStatus(val label: String, val burmeseLabel: String, val colorHex: Long) {
    MERGED("Merged", "တူ", 0xFF00E676),      // Green
    UNCERTAIN("Uncertain", "သံသယ", 0xFFFFD600), // Yellow/Amber
    SPLIT("Split", "ကွဲ", 0xFFFF1744)          // Red
}

data class MergeVerdictResult(
    val pair: Pair<GroupLabel, GroupLabel>,
    val bestFeature: FeatureType,
    val bestCv: Float,
    val pairCvs: Map<String, Float>,
    val isTightest: Boolean,
    val status: MergeVerdictStatus
)

enum class SplitVerdictStatus(val label: String, val burmeseLabel: String, val colorHex: Long) {
    SPLIT("Split", "ကွဲ", 0xFFFF1744),         // Red
    BORDERLINE("Borderline", "သံသယ", 0xFFFFD600), // Yellow
    CLOSE("Close", "မကွဲ / တူ", 0xFF00E676)       // Green
}

data class SplitVerdictResult(
    val comparisonLabel: String,
    val bestFeature: FeatureType,
    val pValue: Double,
    val looPercent: Float,
    val totalScannedTests: Int,
    val status: SplitVerdictStatus
) {
    val bonferroniAlpha: Double
        get() = 0.05 / (totalScannedTests.coerceAtLeast(1) * FeatureType.entries.size)
}

data class SingleValueVerdictResult(
    val selectedFeature: FeatureType,
    val splitComparison: String,
    val leftGroupLabel: String,
    val rightGroupLabel: String,
    val leftMean: Float,
    val rightMean: Float,
    val zGap: Float,
    val pValue: Double,
    val isSignificant: Boolean
)
