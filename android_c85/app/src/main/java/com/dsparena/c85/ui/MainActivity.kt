package com.dsparena.c85.ui

import android.Manifest
import android.content.pm.PackageManager
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.animation.*
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.core.content.ContextCompat
import com.dsparena.c85.ui.components.DotVisualizer
import com.dsparena.c85.ui.model.*
import com.dsparena.c85.ui.screens.TVVerdictScreen
import com.dsparena.c85.ui.theme.*

enum class AppNavTab(val label: String, val icon: androidx.compose.ui.graphics.vector.ImageVector) {
    TV_VERDICTS("TV Verdict", Icons.Default.Analytics),
    FOUR_DOTS_SQUARE("4DS Visualizer", Icons.Default.ScatterPlot)
}

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContent {
            C85DSPTheme {
                MainAppScreen()
            }
        }
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun MainAppScreen() {
    var hasAudioPermission by remember {
        mutableStateOf(false)
    }

    val permissionLauncher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.RequestPermission()
    ) { isGranted ->
        hasAudioPermission = isGranted
    }

    val context = androidx.compose.ui.platform.LocalContext.current
    LaunchedEffect(Unit) {
        hasAudioPermission = ContextCompat.checkSelfPermission(
            context,
            Manifest.permission.RECORD_AUDIO
        ) == PackageManager.PERMISSION_GRANTED
    }

    // App Navigation State
    var currentTab by remember { mutableStateOf(AppNavTab.TV_VERDICTS) }

    // Server WebSocket & Connection State
    var serverHost by remember { mutableStateOf("10.0.2.2:8125") }
    var isConnected by remember { mutableStateOf(false) }
    var isRecording by remember { mutableStateOf(false) }

    // Live DSP & Visualizer State
    var currentRms by remember { mutableFloatStateOf(0.12f) }
    var peakRms by remember { mutableFloatStateOf(0.45f) }
    var isTriggerActive by remember { mutableStateOf(false) }
    var selectedPoint by remember { mutableStateOf<DotPoint?>(null) }

    // Seed/Live Dot Points for 4DS (Groups A, B, C, D)
    val points = remember {
        mutableStateListOf(
            // Class A points
            DotPoint("A-01", GroupLabel.A, 0.22f, 0.78f, mapOf("impact" to 0.78f, "duration" to 495f, "peak_time" to 14f, "mel_mean_db" to -18.2f, "impact_freq_hz" to 3120f)),
            DotPoint("A-02", GroupLabel.A, 0.25f, 0.82f, mapOf("impact" to 0.82f, "duration" to 502f, "peak_time" to 15f, "mel_mean_db" to -17.9f, "impact_freq_hz" to 3180f)),
            DotPoint("A-03", GroupLabel.A, 0.20f, 0.74f, mapOf("impact" to 0.75f, "duration" to 490f, "peak_time" to 13f, "mel_mean_db" to -18.8f, "impact_freq_hz" to 3100f)),
            DotPoint("A-04", GroupLabel.A, 0.24f, 0.80f, mapOf("impact" to 0.80f, "duration" to 498f, "peak_time" to 14f, "mel_mean_db" to -18.0f, "impact_freq_hz" to 3150f)),
            DotPoint("A-05", GroupLabel.A, 0.23f, 0.76f, mapOf("impact" to 0.77f, "duration" to 493f, "peak_time" to 14f, "mel_mean_db" to -18.5f, "impact_freq_hz" to 3130f)),
            DotPoint("A-06", GroupLabel.A, 0.21f, 0.79f, mapOf("impact" to 0.79f, "duration" to 496f, "peak_time" to 14f, "mel_mean_db" to -18.1f, "impact_freq_hz" to 3140f)),

            // Class B points
            DotPoint("B-01", GroupLabel.B, 0.75f, 0.80f, mapOf("impact" to 0.85f, "duration" to 510f, "peak_time" to 16f, "mel_mean_db" to -19.5f, "impact_freq_hz" to 3250f)),
            DotPoint("B-02", GroupLabel.B, 0.78f, 0.84f, mapOf("impact" to 0.88f, "duration" to 518f, "peak_time" to 17f, "mel_mean_db" to -19.1f, "impact_freq_hz" to 3290f)),
            DotPoint("B-03", GroupLabel.B, 0.72f, 0.76f, mapOf("impact" to 0.81f, "duration" to 505f, "peak_time" to 15f, "mel_mean_db" to -19.8f, "impact_freq_hz" to 3210f)),
            DotPoint("B-04", GroupLabel.B, 0.77f, 0.82f, mapOf("impact" to 0.86f, "duration" to 512f, "peak_time" to 16f, "mel_mean_db" to -19.3f, "impact_freq_hz" to 3260f)),
            DotPoint("B-05", GroupLabel.B, 0.74f, 0.79f, mapOf("impact" to 0.83f, "duration" to 508f, "peak_time" to 16f, "mel_mean_db" to -19.6f, "impact_freq_hz" to 3230f)),
            DotPoint("B-06", GroupLabel.B, 0.76f, 0.81f, mapOf("impact" to 0.84f, "duration" to 511f, "peak_time" to 16f, "mel_mean_db" to -19.4f, "impact_freq_hz" to 3240f)),

            // Class C points
            DotPoint("C-01", GroupLabel.C, 0.25f, 0.28f, mapOf("impact" to 0.52f, "duration" to 445f, "peak_time" to 18f, "mel_mean_db" to -24.2f, "impact_freq_hz" to 2150f)),
            DotPoint("C-02", GroupLabel.C, 0.28f, 0.32f, mapOf("impact" to 0.55f, "duration" to 452f, "peak_time" to 19f, "mel_mean_db" to -23.8f, "impact_freq_hz" to 2210f)),
            DotPoint("C-03", GroupLabel.C, 0.22f, 0.25f, mapOf("impact" to 0.49f, "duration" to 440f, "peak_time" to 17f, "mel_mean_db" to -24.6f, "impact_freq_hz" to 2100f)),
            DotPoint("C-04", GroupLabel.C, 0.26f, 0.30f, mapOf("impact" to 0.53f, "duration" to 448f, "peak_time" to 18f, "mel_mean_db" to -24.0f, "impact_freq_hz" to 2180f)),
            DotPoint("C-05", GroupLabel.C, 0.24f, 0.27f, mapOf("impact" to 0.51f, "duration" to 443f, "peak_time" to 18f, "mel_mean_db" to -24.4f, "impact_freq_hz" to 2140f)),
            DotPoint("C-06", GroupLabel.C, 0.27f, 0.29f, mapOf("impact" to 0.54f, "duration" to 447f, "peak_time" to 18f, "mel_mean_db" to -24.1f, "impact_freq_hz" to 2170f)),

            // Class D points
            DotPoint("D-01", GroupLabel.D, 0.74f, 0.26f, mapOf("impact" to 0.56f, "duration" to 455f, "peak_time" to 19f, "mel_mean_db" to -25.5f, "impact_freq_hz" to 2220f)),
            DotPoint("D-02", GroupLabel.D, 0.77f, 0.30f, mapOf("impact" to 0.59f, "duration" to 462f, "peak_time" to 20f, "mel_mean_db" to -25.1f, "impact_freq_hz" to 2280f)),
            DotPoint("D-03", GroupLabel.D, 0.71f, 0.23f, mapOf("impact" to 0.53f, "duration" to 450f, "peak_time" to 18f, "mel_mean_db" to -25.9f, "impact_freq_hz" to 2170f)),
            DotPoint("D-04", GroupLabel.D, 0.75f, 0.28f, mapOf("impact" to 0.57f, "duration" to 458f, "peak_time" to 19f, "mel_mean_db" to -25.3f, "impact_freq_hz" to 2250f)),
            DotPoint("D-05", GroupLabel.D, 0.73f, 0.25f, mapOf("impact" to 0.55f, "duration" to 453f, "peak_time" to 19f, "mel_mean_db" to -25.7f, "impact_freq_hz" to 2200f)),
            DotPoint("D-06", GroupLabel.D, 0.76f, 0.27f, mapOf("impact" to 0.58f, "duration" to 457f, "peak_time" to 19f, "mel_mean_db" to -25.4f, "impact_freq_hz" to 2240f))
        )
    }

    // Honesty Metadata
    val honestyMetadata = remember(points) {
        val counts = GroupLabel.entries.associateWith { grp -> points.count { it.group == grp } }
        HonestyMetadata(
            countsPerGroup = counts,
            source = VerdictSource.WAV_REAL,
            sessionRange = "249-268",
            isInferredSession = true,
            isCrossSession = false,
            sampleRateHz = 44100,
            fmaxHz = 8000,
            topDb = 30f,
            windowSec = 0.3f,
            nFft = 1024,
            hopLength = 256,
            honestCeilingPercent = 86f
        )
    }

    Scaffold(
        topBar = {
            TopAppBar(
                title = {
                    Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                        Text("C85 DSP Arena", fontWeight = FontWeight.Bold, fontSize = 18.sp)
                        Surface(
                            shape = RoundedCornerShape(4.dp),
                            color = PrimaryNeon.copy(alpha = 0.2f)
                        ) {
                            Text(
                                "4DS v1.0",
                                modifier = Modifier.padding(horizontal = 6.dp, vertical = 2.dp),
                                color = PrimaryNeon,
                                fontSize = 10.sp,
                                fontWeight = FontWeight.SemiBold
                            )
                        }
                    }
                },
                actions = {
                    // Connection Status Pill
                    Surface(
                        shape = RoundedCornerShape(12.dp),
                        color = if (isConnected) VerdictGreen.copy(alpha = 0.2f) else VerdictAmber.copy(alpha = 0.2f),
                        border = androidx.compose.foundation.BorderStroke(
                            1.dp,
                            if (isConnected) VerdictGreen else VerdictAmber
                        ),
                        modifier = Modifier.padding(end = 8.dp)
                    ) {
                        Row(
                            modifier = Modifier.padding(horizontal = 8.dp, vertical = 4.dp),
                            verticalAlignment = Alignment.CenterVertically,
                            horizontalArrangement = Arrangement.spacedBy(4.dp)
                        ) {
                            Box(
                                modifier = Modifier
                                    .size(6.dp)
                                    .clip(CircleShape)
                                    .background(if (isConnected) VerdictGreen else VerdictAmber)
                            )
                            Text(
                                text = if (isConnected) "WS CONNECTED" else "OFFLINE",
                                fontSize = 10.sp,
                                fontWeight = FontWeight.Bold,
                                color = if (isConnected) VerdictGreen else VerdictAmber
                            )
                        }
                    }
                },
                colors = TopAppBarDefaults.topAppBarColors(
                    containerColor = SurfaceDark,
                    titleContentColor = TextPrimary
                )
            )
        },
        bottomBar = {
            NavigationBar(
                containerColor = SurfaceDark,
                contentColor = TextSecondary
            ) {
                AppNavTab.entries.forEach { tab ->
                    NavigationBarItem(
                        selected = currentTab == tab,
                        onClick = { currentTab = tab },
                        icon = { Icon(tab.icon, contentDescription = tab.label) },
                        label = { Text(tab.label, fontSize = 11.sp, fontWeight = if (currentTab == tab) FontWeight.Bold else FontWeight.Normal) },
                        colors = NavigationBarItemDefaults.colors(
                            selectedIconColor = PrimaryNeon,
                            selectedTextColor = PrimaryNeon,
                            indicatorColor = SurfaceVariantDark
                        )
                    )
                }
            }
        }
    ) { innerPadding ->
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(innerPadding)
                .background(DarkBackground)
        ) {
            // Server Connection & Mic Control Bar
            Card(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(horizontal = 12.dp, vertical = 8.dp),
                shape = RoundedCornerShape(10.dp),
                colors = CardDefaults.cardColors(containerColor = SurfaceDark)
            ) {
                Column(modifier = Modifier.padding(10.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
                    // WebSocket URL row
                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        verticalAlignment = Alignment.CenterVertically,
                        horizontalArrangement = Arrangement.spacedBy(8.dp)
                    ) {
                        OutlinedTextField(
                            value = serverHost,
                            onValueChange = { serverHost = it },
                            label = { Text("Server WS (ws://<host>:8125)", fontSize = 10.sp) },
                            singleLine = true,
                            modifier = Modifier.weight(1f),
                            textStyle = androidx.compose.ui.text.TextStyle(
                                fontFamily = FontFamily.Monospace,
                                fontSize = 12.sp,
                                color = TextPrimary
                            ),
                            colors = OutlinedTextFieldDefaults.colors(
                                focusedBorderColor = PrimaryNeon,
                                unfocusedBorderColor = OutlineDark
                            )
                        )

                        Button(
                            onClick = { isConnected = !isConnected },
                            shape = RoundedCornerShape(8.dp),
                            colors = ButtonDefaults.buttonColors(
                                containerColor = if (isConnected) VerdictRed else PrimaryNeon,
                                contentColor = if (isConnected) Color.White else OnPrimaryNeon
                            )
                        ) {
                            Text(
                                text = if (isConnected) "Disconnect" else "Connect",
                                fontSize = 12.sp,
                                fontWeight = FontWeight.Bold
                            )
                        }
                    }

                    // Audio Recording & Permission Row
                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        horizontalArrangement = Arrangement.SpaceBetween,
                        verticalAlignment = Alignment.CenterVertically
                    ) {
                        if (!hasAudioPermission) {
                            Button(
                                onClick = { permissionLauncher.launch(Manifest.permission.RECORD_AUDIO) },
                                colors = ButtonDefaults.buttonColors(containerColor = VerdictAmber, contentColor = Color.Black),
                                shape = RoundedCornerShape(8.dp)
                            ) {
                                Icon(Icons.Default.Mic, contentDescription = null, modifier = Modifier.size(16.dp))
                                Spacer(Modifier.width(6.dp))
                                Text("Grant Mic Permission", fontSize = 12.sp, fontWeight = FontWeight.Bold)
                            }
                        } else {
                            Button(
                                onClick = {
                                    isRecording = !isRecording
                                    if (isRecording) {
                                        currentRms = 0.28f
                                        peakRms = 0.65f
                                        isTriggerActive = true
                                    } else {
                                        currentRms = 0.05f
                                        peakRms = 0.15f
                                        isTriggerActive = false
                                    }
                                },
                                shape = RoundedCornerShape(8.dp),
                                colors = ButtonDefaults.buttonColors(
                                    containerColor = if (isRecording) VerdictRed else VerdictGreen,
                                    contentColor = Color.White
                                )
                            ) {
                                Icon(
                                    if (isRecording) Icons.Default.Stop else Icons.Default.Mic,
                                    contentDescription = null,
                                    modifier = Modifier.size(16.dp)
                                )
                                Spacer(Modifier.width(6.dp))
                                Text(
                                    text = if (isRecording) "Stop 44.1k Rec" else "Start 44.1k Rec",
                                    fontSize = 12.sp,
                                    fontWeight = FontWeight.Bold
                                )
                            }
                        }

                        Text(
                            text = if (isRecording) "● REC (44.1kHz 16b Mono)" else "STANDBY",
                            fontFamily = FontFamily.Monospace,
                            fontSize = 11.sp,
                            color = if (isRecording) VerdictRed else TextMuted,
                            fontWeight = FontWeight.Bold
                        )
                    }
                }
            }

            // Tab View Body
            when (currentTab) {
                AppNavTab.TV_VERDICTS -> {
                    TVVerdictScreen(
                        metadata = honestyMetadata,
                        onMakeSimilarAction = { pair ->
                            // Hill-climb simulation: attracts pair dots in 4DS
                            val g1 = pair.first
                            val g2 = pair.second
                            // Small feedback trigger
                        },
                        onMakeDifferentAction = { splitLabel ->
                            // Hill-climb simulation: splits groups
                        }
                    )
                }
                AppNavTab.FOUR_DOTS_SQUARE -> {
                    DotVisualizer(
                        points = points,
                        selectedPoint = selectedPoint,
                        onPointSelect = { selectedPoint = it },
                        currentRmsLevel = currentRms,
                        peakRmsLevel = peakRms,
                        triggerThreshold = 0.35f,
                        isTriggerActive = isTriggerActive
                    )
                }
            }
        }
    }
}
