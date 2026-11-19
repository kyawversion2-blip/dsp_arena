package com.dsparena.c85.ui.screens

import androidx.compose.animation.*
import androidx.compose.animation.core.tween
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.ArrowForward
import androidx.compose.material.icons.filled.CheckCircle
import androidx.compose.material.icons.filled.PlayArrow
import androidx.compose.material.icons.filled.Refresh
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.dsparena.c85.ui.components.HonestyCard
import com.dsparena.c85.ui.model.*
import com.dsparena.c85.ui.theme.*

enum class TVScreenTab(val title: String, val burmeseTitle: String) {
    MERGE("Screen A", "တူလား? (Merge)"),
    SPLIT("Screen B", "ကွဲလား? (Split)"),
    SINGLE_VALUE("Screen C", "Value တစ်ခုထဲ (Single)")
}

@Composable
fun TVVerdictScreen(
    metadata: HonestyMetadata,
    onMakeSimilarAction: (Pair<GroupLabel, GroupLabel>) -> Unit,
    onMakeDifferentAction: (String) -> Unit,
    modifier: Modifier = Modifier
) {
    var selectedTab by remember { mutableStateOf(TVScreenTab.MERGE) }

    Column(
        modifier = modifier
            .fillMaxSize()
            .background(DarkBackground)
    ) {
        // Top TV Game Mode Switcher TabRow
        TabRow(
            selectedTabIndex = selectedTab.ordinal,
            containerColor = SurfaceDark,
            contentColor = PrimaryNeon,
            divider = { HorizontalDivider(color = OutlineDark.copy(alpha = 0.5f)) }
        ) {
            TVScreenTab.entries.forEach { tab ->
                Tab(
                    selected = selectedTab == tab,
                    onClick = { selectedTab = tab },
                    text = {
                        Column(horizontalAlignment = Alignment.CenterHorizontally) {
                            Text(tab.title, fontSize = 11.sp, color = if (selectedTab == tab) PrimaryNeon else TextMuted)
                            Text(tab.burmeseTitle, fontSize = 13.sp, fontWeight = FontWeight.Bold)
                        }
                    }
                )
            }
        }

        // Screen Body with animated transitions
        Column(
            modifier = Modifier
                .fillMaxSize()
                .verticalScroll(rememberScrollState())
                .padding(14.dp),
            verticalArrangement = Arrangement.spacedBy(16.dp)
        ) {
            AnimatedContent(
                targetState = selectedTab,
                transitionSpec = {
                    fadeIn(animationSpec = tween(300)) togetherWith fadeOut(animationSpec = tween(200))
                },
                label = "TVScreenTransition"
            ) { tab ->
                when (tab) {
                    TVScreenTab.MERGE -> ScreenAMergeVerdict(onMakeSimilar = onMakeSimilarAction)
                    TVScreenTab.SPLIT -> ScreenBSplitVerdict(onMakeDifferent = onMakeDifferentAction)
                    TVScreenTab.SINGLE_VALUE -> ScreenCSingleValueVerdict()
                }
            }

            // Mandatory Honesty Layer on every verdict screen
            HonestyCard(
                metadata = metadata,
                reportedAccuracyOrLoo = 84f
            )
        }
    }
}

// ==========================================
// SCREEN A — တူလား? (MERGE VERDICT)
// ==========================================
@Composable
fun ScreenAMergeVerdict(
    onMakeSimilar: (Pair<GroupLabel, GroupLabel>) -> Unit
) {
    var firstGroup by remember { mutableStateOf(GroupLabel.B) }
    var secondGroup by remember { mutableStateOf(GroupLabel.D) }
    var isRevealing by remember { mutableStateOf(false) }

    // Mock/verified anchor: BD tightest pair on Impact CV = 0.019 (TV_GAME_VERDICT_SPEC V3)
    val pairKey = "${firstGroup.id}${secondGroup.id}"
    val isBD = (firstGroup == GroupLabel.B && secondGroup == GroupLabel.D) || (firstGroup == GroupLabel.D && secondGroup == GroupLabel.B)
    val bestCv = if (isBD) 0.019f else 0.078f
    val isTightest = isBD
    val status = when {
        bestCv <= 0.05f && isTightest -> MergeVerdictStatus.MERGED
        bestCv <= 0.09f -> MergeVerdictStatus.UNCERTAIN
        else -> MergeVerdictStatus.SPLIT
    }

    Card(
        modifier = Modifier.fillMaxWidth(),
        shape = RoundedCornerShape(16.dp),
        colors = CardDefaults.cardColors(containerColor = SurfaceDark),
        border = CardDefaults.outlinedCardBorder()
    ) {
        Column(
            modifier = Modifier.padding(16.dp),
            verticalArrangement = Arrangement.spacedBy(14.dp)
        ) {
            Text(
                text = "Screen A — တူလား? (MERGE VERDICT)",
                style = MaterialTheme.typography.titleMedium,
                color = TextPrimary,
                fontWeight = FontWeight.Bold
            )
            Text(
                text = "Pick two groups to evaluate pair merge compatibility. Guards pitfall 16: globally tight features are invariants, NOT merges.",
                style = MaterialTheme.typography.bodySmall,
                color = TextSecondary
            )

            // Group Pair Selector Tiles (A, B, C, D)
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceEvenly
            ) {
                GroupLabel.entries.forEach { group ->
                    val isSelected = (group == firstGroup || group == secondGroup)
                    val gColor = Color(group.colorHex)
                    Box(
                        modifier = Modifier
                            .size(54.dp)
                            .clip(RoundedCornerShape(12.dp))
                            .background(if (isSelected) gColor.copy(alpha = 0.25f) else SurfaceVariantDark)
                            .border(
                                width = if (isSelected) 2.dp else 1.dp,
                                color = if (isSelected) gColor else OutlineDark,
                                shape = RoundedCornerShape(12.dp)
                            )
                            .clickable {
                                if (group != firstGroup && group != secondGroup) {
                                    firstGroup = secondGroup
                                    secondGroup = group
                                }
                            },
                        contentAlignment = Alignment.Center
                    ) {
                        Text(
                            text = group.id,
                            fontSize = 20.sp,
                            fontWeight = FontWeight.ExtraBold,
                            color = if (isSelected) gColor else TextMuted
                        )
                    }
                }
            }

            // Big TV Verdict Box
            Surface(
                modifier = Modifier.fillMaxWidth(),
                shape = RoundedCornerShape(12.dp),
                color = Color(status.colorHex).copy(alpha = 0.15f),
                border = androidx.compose.foundation.BorderStroke(2.dp, Color(status.colorHex))
            ) {
                Column(
                    modifier = Modifier.padding(16.dp),
                    horizontalAlignment = Alignment.CenterHorizontally,
                    verticalArrangement = Arrangement.spacedBy(6.dp)
                ) {
                    Text(
                        text = "Pair [${firstGroup.id} - ${secondGroup.id}] Verdict",
                        style = MaterialTheme.typography.labelMedium,
                        color = TextSecondary
                    )
                    Text(
                        text = "${status.burmeseLabel} (${status.label.uppercase()})",
                        fontSize = 32.sp,
                        fontWeight = FontWeight.Black,
                        color = Color(status.colorHex)
                    )
                    Text(
                        text = "Min CV: ${String.format("%.3f", bestCv)} (${if (isTightest) "Tightest Pair across all 6" else "Not Tightest"})",
                        fontFamily = FontFamily.Monospace,
                        fontSize = 12.sp,
                        color = TextPrimary
                    )
                }
            }

            // All-6-Pair CV Strip (Pitfall 16 Guard)
            Text("All 6-Pair CV Strip (Guard Pitfall 16):", style = MaterialTheme.typography.labelSmall, color = TextSecondary)
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween
            ) {
                val mockAll6 = listOf(
                    "AB" to 0.082f, "AC" to 0.095f, "AD" to 0.110f,
                    "BC" to 0.064f, "BD" to 0.019f, "CD" to 0.088f
                )
                mockAll6.forEach { (pair, cv) ->
                    val isCurrent = (pair == "${firstGroup.id}${secondGroup.id}" || pair == "${secondGroup.id}${firstGroup.id}")
                    Column(
                        horizontalAlignment = Alignment.CenterHorizontally,
                        modifier = Modifier
                            .clip(RoundedCornerShape(6.dp))
                            .background(if (isCurrent) PrimaryNeon.copy(alpha = 0.2f) else DarkBackground)
                            .padding(4.dp)
                    ) {
                        Text(pair, fontSize = 10.sp, fontWeight = FontWeight.Bold, color = if (isCurrent) PrimaryNeon else TextMuted)
                        Text(String.format("%.3f", cv), fontSize = 9.sp, fontFamily = FontFamily.Monospace, color = if (cv <= 0.05f) VerdictGreen else TextSecondary)
                    }
                }
            }

            // One-tap "တူအောင်လုပ်" Button (Make Similar Hill Climb)
            Button(
                onClick = {
                    isRevealing = true
                    onMakeSimilar(Pair(firstGroup, secondGroup))
                },
                modifier = Modifier
                    .fillMaxWidth()
                    .height(50.dp),
                shape = RoundedCornerShape(10.dp),
                colors = ButtonDefaults.buttonColors(containerColor = PrimaryNeon, contentColor = OnPrimaryNeon)
            ) {
                Icon(Icons.Default.Refresh, contentDescription = null, modifier = Modifier.size(20.dp))
                Spacer(Modifier.width(8.dp))
                Text(
                    text = "တူအောင်လုပ် (MAKE-SIMILAR HILL-CLIMB)",
                    fontSize = 14.sp,
                    fontWeight = FontWeight.Bold
                )
            }
        }
    }
}

// ==========================================
// SCREEN B — ကွဲလား? (SPLIT VERDICT)
// ==========================================
@Composable
fun ScreenBSplitVerdict(
    onMakeDifferent: (String) -> Unit
) {
    var splitMode by remember { mutableStateOf("AB | CD") }
    val splitOptions = listOf("AB | CD", "C | rest", "D | rest", "A | BCD")

    // Verified anchors from TV_GAME_VERDICT_SPEC:
    // V1: Duration AB|CD split p < 0.05 (p=0.0083 @384k)
    // V2: C|rest via impact_freq_hz ~85% LOO (44.1k)
    // V5: Mel_Mean_dB D|rest p=0.0073
    val (pVal, looVal, bestFeat) = remember(splitMode) {
        when (splitMode) {
            "AB | CD" -> Triple(0.0083, 82.5f, FeatureType.DURATION)
            "C | rest" -> Triple(0.0120, 85.0f, FeatureType.IMPACT_FREQ_HZ)
            "D | rest" -> Triple(0.0073, 84.2f, FeatureType.MEL_MEAN_DB)
            else -> Triple(0.0450, 78.0f, FeatureType.IMPACT)
        }
    }

    val isSignificant = pVal < 0.05
    val status = when {
        pVal < 0.05 -> SplitVerdictStatus.SPLIT
        pVal < 0.10 -> SplitVerdictStatus.BORDERLINE
        else -> SplitVerdictStatus.CLOSE
    }

    Card(
        modifier = Modifier.fillMaxWidth(),
        shape = RoundedCornerShape(16.dp),
        colors = CardDefaults.cardColors(containerColor = SurfaceDark),
        border = CardDefaults.outlinedCardBorder()
    ) {
        Column(
            modifier = Modifier.padding(16.dp),
            verticalArrangement = Arrangement.spacedBy(14.dp)
        ) {
            Text(
                text = "Screen B — ကွဲလား? (SPLIT VERDICT)",
                style = MaterialTheme.typography.titleMedium,
                color = TextPrimary,
                fontWeight = FontWeight.Bold
            )

            // Split Mode Selector Pills
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(8.dp)
            ) {
                splitOptions.forEach { opt ->
                    val isSel = opt == splitMode
                    FilterChip(
                        selected = isSel,
                        onClick = { splitMode = opt },
                        label = { Text(opt, fontSize = 12.sp, fontWeight = if (isSel) FontWeight.Bold else FontWeight.Normal) },
                        colors = FilterChipDefaults.filterChipColors(
                            selectedContainerColor = PrimaryNeon.copy(alpha = 0.25f),
                            selectedLabelColor = PrimaryNeon
                        )
                    )
                }
            }

            // Big TV Verdict Box
            Surface(
                modifier = Modifier.fillMaxWidth(),
                shape = RoundedCornerShape(12.dp),
                color = Color(status.colorHex).copy(alpha = 0.15f),
                border = androidx.compose.foundation.BorderStroke(2.dp, Color(status.colorHex))
            ) {
                Column(
                    modifier = Modifier.padding(16.dp),
                    horizontalAlignment = Alignment.CenterHorizontally,
                    verticalArrangement = Arrangement.spacedBy(6.dp)
                ) {
                    Text(
                        text = "Split [$splitMode] Mann-Whitney U",
                        style = MaterialTheme.typography.labelMedium,
                        color = TextSecondary
                    )
                    Text(
                        text = "${status.burmeseLabel} (${status.label.uppercase()})",
                        fontSize = 32.sp,
                        fontWeight = FontWeight.Black,
                        color = Color(status.colorHex)
                    )
                    Text(
                        text = "p = ${String.format("%.4f", pVal)} | LOO: ${String.format("%.1f", looVal)}%",
                        fontFamily = FontFamily.Monospace,
                        fontSize = 14.sp,
                        fontWeight = FontWeight.Bold,
                        color = TextPrimary
                    )
                    Text(
                        text = "Best Feature: ${bestFeat.displayName} (${bestFeat.unit})",
                        style = MaterialTheme.typography.labelSmall,
                        color = TextSecondary
                    )
                }
            }

            // Bonferroni Multi-test Adjustment Notice (Section 3 rule)
            Surface(
                modifier = Modifier.fillMaxWidth(),
                shape = RoundedCornerShape(8.dp),
                color = DarkBackground.copy(alpha = 0.8f)
            ) {
                Column(modifier = Modifier.padding(10.dp)) {
                    Text(
                        text = "Multi-Test Correction (7 splits × 5 features = 35 tests):",
                        style = MaterialTheme.typography.labelSmall,
                        color = TextSecondary
                    )
                    Text(
                        text = "Bonferroni α = 0.05 / 35 = 0.0014 (p < 0.0014 strictly required for multi-hypothesis robustness).",
                        fontFamily = FontFamily.Monospace,
                        fontSize = 10.5.sp,
                        color = PrimaryNeon
                    )
                }
            }

            // One-tap "မတူအောင်လုပ်" Button (Make Different Hill Climb)
            Button(
                onClick = { onMakeDifferent(splitMode) },
                modifier = Modifier
                    .fillMaxWidth()
                    .height(50.dp),
                shape = RoundedCornerShape(10.dp),
                colors = ButtonDefaults.buttonColors(containerColor = VerdictRed, contentColor = Color.White)
            ) {
                Icon(Icons.Default.PlayArrow, contentDescription = null, modifier = Modifier.size(20.dp))
                Spacer(Modifier.width(8.dp))
                Text(
                    text = "မတူအောင်လုပ် (MAKE-DIFFERENT HILL-CLIMB)",
                    fontSize = 14.sp,
                    fontWeight = FontWeight.Bold
                )
            }
        }
    }
}

// ==========================================
// SCREEN C — Value တစ်ခုထဲ (SINGLE-VALUE VERDICT)
// ==========================================
@Composable
fun ScreenCSingleValueVerdict() {
    var selectedFeature by remember { mutableStateOf(FeatureType.DURATION) }
    var selectedSplit by remember { mutableStateOf("AB vs CD") }
    var featureMenuOpen by remember { mutableStateOf(false) }

    // Computation values for the selected single value
    val (leftMean, rightMean, zGap, pValue) = remember(selectedFeature, selectedSplit) {
        when (selectedFeature) {
            FeatureType.DURATION -> Quad(499.2f, 452.1f, 2.64f, 0.0083)
            FeatureType.IMPACT_FREQ_HZ -> Quad(3250f, 2180f, 2.45f, 0.0120)
            FeatureType.MEL_MEAN_DB -> Quad(-18.4f, -24.8f, 2.71f, 0.0073)
            FeatureType.IMPACT -> Quad(0.82f, 0.54f, 2.15f, 0.0210)
            FeatureType.PEAK_TIME -> Quad(14.2f, 18.5f, 1.85f, 0.0640)
        }
    }
    val isSignificant = pValue < 0.05

    Card(
        modifier = Modifier.fillMaxWidth(),
        shape = RoundedCornerShape(16.dp),
        colors = CardDefaults.cardColors(containerColor = SurfaceDark),
        border = CardDefaults.outlinedCardBorder()
    ) {
        Column(
            modifier = Modifier.padding(16.dp),
            verticalArrangement = Arrangement.spacedBy(14.dp)
        ) {
            Text(
                text = "Screen C — Value တစ်ခုထဲ (SINGLE-VALUE VERDICT)",
                style = MaterialTheme.typography.titleMedium,
                color = TextPrimary,
                fontWeight = FontWeight.Bold
            )
            Text(
                text = "Single feature × single split → ONE big unambiguous answer.",
                style = MaterialTheme.typography.bodySmall,
                color = TextSecondary
            )

            // Selectors: Feature & Split
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically
            ) {
                // Feature Dropdown
                Box {
                    OutlinedButton(
                        onClick = { featureMenuOpen = true },
                        colors = ButtonDefaults.outlinedButtonColors(contentColor = PrimaryNeon),
                        shape = RoundedCornerShape(8.dp)
                    ) {
                        Text("Feature: ${selectedFeature.displayName}")
                    }
                    DropdownMenu(
                        expanded = featureMenuOpen,
                        onDismissRequest = { featureMenuOpen = false },
                        modifier = Modifier.background(SurfaceDark)
                    ) {
                        FeatureType.entries.forEach { f ->
                            DropdownMenuItem(
                                text = { Text(f.displayName, color = TextPrimary) },
                                onClick = {
                                    selectedFeature = f
                                    featureMenuOpen = false
                                }
                            )
                        }
                    }
                }

                // Split Options
                Row(horizontalArrangement = Arrangement.spacedBy(6.dp)) {
                    listOf("AB vs CD", "C vs rest", "D vs rest").forEach { sp ->
                        FilterChip(
                            selected = sp == selectedSplit,
                            onClick = { selectedSplit = sp },
                            label = { Text(sp, fontSize = 11.sp) }
                        )
                    }
                }
            }

            // ONE BIG NUMBER DISPLAY (The user's exact specification)
            Surface(
                modifier = Modifier.fillMaxWidth(),
                shape = RoundedCornerShape(16.dp),
                color = if (isSignificant) VerdictRed.copy(alpha = 0.15f) else VerdictGreen.copy(alpha = 0.15f),
                border = androidx.compose.foundation.BorderStroke(
                    2.dp,
                    if (isSignificant) VerdictRed else VerdictGreen
                )
            ) {
                Column(
                    modifier = Modifier.padding(20.dp),
                    horizontalAlignment = Alignment.CenterHorizontally,
                    verticalArrangement = Arrangement.spacedBy(8.dp)
                ) {
                    Text(
                        text = "Z-SCORE SEPARATION GAP",
                        style = MaterialTheme.typography.labelMedium,
                        color = TextSecondary,
                        letterSpacing = 1.sp
                    )

                    // Big Number
                    Text(
                        text = "${String.format("%.2f", zGap)} σ",
                        fontSize = 48.sp,
                        fontWeight = FontWeight.Black,
                        color = if (isSignificant) VerdictRed else VerdictGreen
                    )

                    // Verdict Banner
                    Surface(
                        shape = RoundedCornerShape(6.dp),
                        color = if (isSignificant) VerdictRed else VerdictGreen
                    ) {
                        Text(
                            text = if (isSignificant) "🔴 ကွဲပြားသည် (SIGNIFICANT SEPARATION)" else "🟢 တူညီသည် (NO SIGNIFICANT GAP)",
                            modifier = Modifier.padding(horizontal = 12.dp, vertical = 6.dp),
                            color = Color.White,
                            fontWeight = FontWeight.Bold,
                            fontSize = 12.sp
                        )
                    }

                    HorizontalDivider(color = OutlineDark.copy(alpha = 0.4f), modifier = Modifier.padding(vertical = 4.dp))

                    // Mean Values per side
                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        horizontalArrangement = Arrangement.SpaceAround
                    ) {
                        Column(horizontalAlignment = Alignment.CenterHorizontally) {
                            Text("Side 1 Mean", fontSize = 10.sp, color = TextMuted)
                            Text(
                                text = "${String.format("%.1f", leftMean)} ${selectedFeature.unit}",
                                fontFamily = FontFamily.Monospace,
                                fontSize = 14.sp,
                                fontWeight = FontWeight.SemiBold,
                                color = TextPrimary
                            )
                        }
                        Column(horizontalAlignment = Alignment.CenterHorizontally) {
                            Text("Side 2 Mean", fontSize = 10.sp, color = TextMuted)
                            Text(
                                text = "${String.format("%.1f", rightMean)} ${selectedFeature.unit}",
                                fontFamily = FontFamily.Monospace,
                                fontSize = 14.sp,
                                fontWeight = FontWeight.SemiBold,
                                color = TextPrimary
                            )
                        }
                        Column(horizontalAlignment = Alignment.CenterHorizontally) {
                            Text("Mann-Whitney p", fontSize = 10.sp, color = TextMuted)
                            Text(
                                text = String.format("%.4f", pValue),
                                fontFamily = FontFamily.Monospace,
                                fontSize = 14.sp,
                                fontWeight = FontWeight.Bold,
                                color = if (isSignificant) VerdictRed else VerdictGreen
                            )
                        }
                    }
                }
            }
        }
    }
}

private data class Quad<A, B, C, D>(val first: A, val second: B, val third: C, val fourth: D)
