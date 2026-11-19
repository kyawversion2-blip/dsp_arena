package com.dsparena.c85.ui.components

import androidx.compose.animation.AnimatedVisibility
import androidx.compose.animation.core.*
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.gestures.detectTapGestures
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.PathEffect
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.dsparena.c85.ui.model.DotPoint
import com.dsparena.c85.ui.model.FeatureType
import com.dsparena.c85.ui.model.GroupLabel
import com.dsparena.c85.ui.theme.*
import kotlin.math.sqrt

@Composable
fun DotVisualizer(
    points: List<DotPoint>,
    selectedPoint: DotPoint?,
    onPointSelect: (DotPoint?) -> Unit,
    currentRmsLevel: Float,
    peakRmsLevel: Float,
    triggerThreshold: Float,
    isTriggerActive: Boolean,
    modifier: Modifier = Modifier
) {
    var featureX by remember { mutableStateOf(FeatureType.IMPACT) }
    var featureY by remember { mutableStateOf(FeatureType.MEL_MEAN_DB) }

    Column(
        modifier = modifier
            .fillMaxWidth()
            .background(DarkBackground)
            .padding(12.dp),
        verticalArrangement = Arrangement.spacedBy(10.dp)
    ) {
        // Top Bar: Audio Level Meter & Impact Detection Indicator
        AudioLevelMeter(
            currentLevel = currentRmsLevel.coerceIn(0f, 1f),
            peakLevel = peakRmsLevel.coerceIn(0f, 1f),
            threshold = triggerThreshold.coerceIn(0f, 1f),
            isTriggerActive = isTriggerActive
        )

        // Feature Axis Selection Row
        Card(
            colors = CardDefaults.cardColors(containerColor = SurfaceDark),
            shape = RoundedCornerShape(8.dp)
        ) {
            Row(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(horizontal = 12.dp, vertical = 6.dp),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically
            ) {
                Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(4.dp)) {
                    Text("X: ", style = MaterialTheme.typography.labelMedium, color = TextSecondary)
                    AxisFeatureSelector(current = featureX, onSelect = { featureX = it })
                }
                Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(4.dp)) {
                    Text("Y: ", style = MaterialTheme.typography.labelMedium, color = TextSecondary)
                    AxisFeatureSelector(current = featureY, onSelect = { featureY = it })
                }
            }
        }

        // 4 Dots Square Interactive 2D Canvas
        FourDotsSquareCanvas(
            points = points,
            selectedPoint = selectedPoint,
            onPointSelect = onPointSelect,
            featureX = featureX,
            featureY = featureY,
            modifier = Modifier
                .fillMaxWidth()
                .aspectRatio(1.0f)
                .border(1.dp, OutlineDark, RoundedCornerShape(12.dp))
        )

        // Selected Point Feature Inspector
        AnimatedVisibility(visible = selectedPoint != null) {
            selectedPoint?.let { point ->
                PointDetailCard(point = point, onClose = { onPointSelect(null) })
            }
        }
    }
}

@Composable
fun AudioLevelMeter(
    currentLevel: Float,
    peakLevel: Float,
    threshold: Float,
    isTriggerActive: Boolean,
    modifier: Modifier = Modifier
) {
    Card(
        modifier = modifier.fillMaxWidth(),
        colors = CardDefaults.cardColors(containerColor = SurfaceDark),
        shape = RoundedCornerShape(8.dp)
    ) {
        Column(modifier = Modifier.padding(10.dp), verticalArrangement = Arrangement.spacedBy(6.dp)) {
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically
            ) {
                Text(
                    text = "MIC AUDIO LEVEL (44.1kHz PCM)",
                    style = MaterialTheme.typography.labelSmall,
                    color = TextSecondary,
                    fontWeight = FontWeight.Bold
                )

                Row(
                    verticalAlignment = Alignment.CenterVertically,
                    horizontalArrangement = Arrangement.spacedBy(6.dp)
                ) {
                    Box(
                        modifier = Modifier
                            .size(10.dp)
                            .background(
                                color = if (isTriggerActive) VerdictRed else TextMuted,
                                shape = RoundedCornerShape(5.dp)
                            )
                    )
                    Text(
                        text = if (isTriggerActive) "IMPACT DETECTED" else "ARMED (1kHz HP)",
                        style = MaterialTheme.typography.labelSmall,
                        color = if (isTriggerActive) VerdictRed else TextMuted,
                        fontWeight = FontWeight.SemiBold
                    )
                }
            }

            // Meter Canvas
            Canvas(
                modifier = Modifier
                    .fillMaxWidth()
                    .height(18.dp)
                    .background(DarkBackground, RoundedCornerShape(4.dp))
            ) {
                val width = size.width
                val height = size.height

                // Active level bar
                val currentWidth = width * currentLevel
                drawRect(
                    brush = Brush.horizontalGradient(
                        colors = listOf(LevelMeterLow, LevelMeterMid, LevelMeterHigh),
                        startX = 0f,
                        endX = width
                    ),
                    topLeft = Offset(0f, 0f),
                    size = Size(currentWidth, height)
                )

                // Peak line
                val peakX = width * peakLevel
                drawLine(
                    color = Color.White,
                    start = Offset(peakX, 0f),
                    end = Offset(peakX, height),
                    strokeWidth = 2.dp.toPx()
                )

                // Trigger threshold line
                val threshX = width * threshold
                drawLine(
                    color = VerdictAmber,
                    start = Offset(threshX, 0f),
                    end = Offset(threshX, height),
                    strokeWidth = 2.dp.toPx(),
                    pathEffect = PathEffect.dashPathEffect(floatArrayOf(6f, 4f), 0f)
                )
            }
        }
    }
}

@Composable
fun AxisFeatureSelector(
    current: FeatureType,
    onSelect: (FeatureType) -> Unit
) {
    var expanded by remember { mutableStateOf(false) }

    Box {
        OutlinedButton(
            onClick = { expanded = true },
            contentPadding = PaddingValues(horizontal = 8.dp, vertical = 2.dp),
            shape = RoundedCornerShape(6.dp),
            colors = ButtonDefaults.outlinedButtonColors(contentColor = PrimaryNeon)
        ) {
            Text(current.displayName, fontSize = 11.sp, fontWeight = FontWeight.SemiBold)
        }
        DropdownMenu(
            expanded = expanded,
            onDismissRequest = { expanded = false },
            modifier = Modifier.background(SurfaceDark)
        ) {
            FeatureType.entries.forEach { feat ->
                DropdownMenuItem(
                    text = { Text(feat.displayName, color = TextPrimary, fontSize = 12.sp) },
                    onClick = {
                        onSelect(feat)
                        expanded = false
                    }
                )
            }
        }
    }
}

@Composable
fun FourDotsSquareCanvas(
    points: List<DotPoint>,
    selectedPoint: DotPoint?,
    onPointSelect: (DotPoint?) -> Unit,
    featureX: FeatureType,
    featureY: FeatureType,
    modifier: Modifier = Modifier
) {
    // Determine bounds for normalization
    val (minX, maxX) = remember(points, featureX) {
        val vals = points.mapNotNull { it.features[featureX.key] }
        if (vals.isEmpty()) 0f to 1f
        else {
            val min = vals.minOrNull() ?: 0f
            val max = vals.maxOrNull() ?: 1f
            if (min == max) (min - 1f) to (max + 1f) else min to max
        }
    }

    val (minY, maxY) = remember(points, featureY) {
        val vals = points.mapNotNull { it.features[featureY.key] }
        if (vals.isEmpty()) 0f to 1f
        else {
            val min = vals.minOrNull() ?: 0f
            val max = vals.maxOrNull() ?: 1f
            if (min == max) (min - 1f) to (max + 1f) else min to max
        }
    }

    // Centroids for A, B, C, D
    val centroids = remember(points, minX, maxX, minY, maxY, featureX, featureY) {
        GroupLabel.entries.mapNotNull { group ->
            val grpPoints = points.filter { it.group == group && it.features.containsKey(featureX.key) && it.features.containsKey(featureY.key) }
            if (grpPoints.isEmpty()) null
            else {
                val avgX = grpPoints.map { it.features[featureX.key]!! }.average().toFloat()
                val avgY = grpPoints.map { it.features[featureY.key]!! }.average().toFloat()
                val normX = ((avgX - minX) / (maxX - minX)).coerceIn(0.05f, 0.95f)
                val normY = 1.0f - ((avgY - minY) / (maxY - minY)).coerceIn(0.05f, 0.95f) // Invert Y for canvas
                group to Offset(normX, normY)
            }
        }.toMap()
    }

    Canvas(
        modifier = modifier
            .background(DarkBackground)
            .pointerInput(points, minX, maxX, minY, maxY, featureX, featureY) {
                detectTapGestures { tapOffset ->
                    val w = size.width
                    val h = size.height
                    var closestPoint: DotPoint? = null
                    var minDistance = Float.MAX_VALUE
                    val touchRadius = 28.dp.toPx()

                    for (p in points) {
                        val fx = p.features[featureX.key] ?: continue
                        val fy = p.features[featureY.key] ?: continue
                        val px = ((fx - minX) / (maxX - minX)).coerceIn(0.05f, 0.95f) * w
                        val py = (1.0f - ((fy - minY) / (maxY - minY)).coerceIn(0.05f, 0.95f)) * h
                        val dist = sqrt((tapOffset.x - px) * (tapOffset.x - px) + (tapOffset.y - py) * (tapOffset.y - py))
                        if (dist < touchRadius && dist < minDistance) {
                            minDistance = dist
                            closestPoint = p
                        }
                    }
                    onPointSelect(closestPoint)
                }
            }
    ) {
        val w = size.width
        val h = size.height
        val padding = 16.dp.toPx()

        // 1. Draw Grid lines and Square boundary
        drawRect(
            color = OutlineDark.copy(alpha = 0.35f),
            topLeft = Offset(padding, padding),
            size = Size(w - 2 * padding, h - 2 * padding),
            style = Stroke(width = 1.dp.toPx())
        )

        // Center crosshairs (dividing 4 quadrants: A, B, C, D)
        val midX = w / 2f
        val midY = h / 2f
        val dashEffect = PathEffect.dashPathEffect(floatArrayOf(8f, 6f), 0f)

        drawLine(
            color = OutlineDark.copy(alpha = 0.4f),
            start = Offset(padding, midY),
            end = Offset(w - padding, midY),
            strokeWidth = 1.dp.toPx(),
            pathEffect = dashEffect
        )
        drawLine(
            color = OutlineDark.copy(alpha = 0.4f),
            start = Offset(midX, padding),
            end = Offset(midX, h - padding),
            strokeWidth = 1.dp.toPx(),
            pathEffect = dashEffect
        )

        // 2. Draw 4DS Quadrant labels in corners
        // (A: top-left, B: top-right, C: bottom-left, D: bottom-right)
        val cornerOffset = 24.dp.toPx()
        // We draw small corner markers
        drawCircle(color = ClassAColor.copy(alpha = 0.2f), radius = 6.dp.toPx(), center = Offset(padding + cornerOffset, padding + cornerOffset))
        drawCircle(color = ClassBColor.copy(alpha = 0.2f), radius = 6.dp.toPx(), center = Offset(w - padding - cornerOffset, padding + cornerOffset))
        drawCircle(color = ClassCColor.copy(alpha = 0.2f), radius = 6.dp.toPx(), center = Offset(padding + cornerOffset, h - padding - cornerOffset))
        drawCircle(color = ClassDColor.copy(alpha = 0.2f), radius = 6.dp.toPx(), center = Offset(w - padding - cornerOffset, h - padding - cornerOffset))

        // 3. Draw Centroid Rings
        centroids.forEach { (group, normOffset) ->
            val center = Offset(
                padding + normOffset.x * (w - 2 * padding),
                padding + normOffset.y * (h - 2 * padding)
            )
            val gColor = Color(group.colorHex)
            drawCircle(
                color = gColor.copy(alpha = 0.25f),
                radius = 16.dp.toPx(),
                center = center
            )
            drawCircle(
                color = gColor,
                radius = 4.dp.toPx(),
                center = center
            )
        }

        // 4. Draw Individual Points
        for (point in points) {
            val fx = point.features[featureX.key] ?: continue
            val fy = point.features[featureY.key] ?: continue
            val nx = ((fx - minX) / (maxX - minX)).coerceIn(0.05f, 0.95f)
            val ny = 1.0f - ((fy - minY) / (maxY - minY)).coerceIn(0.05f, 0.95f)

            val px = padding + nx * (w - 2 * padding)
            val py = padding + ny * (h - 2 * padding)
            val pointColor = Color(point.group.colorHex)
            val isSelected = selectedPoint?.id == point.id

            if (isSelected) {
                // Outer glow ring
                drawCircle(
                    color = Color.White,
                    radius = 12.dp.toPx(),
                    center = Offset(px, py),
                    style = Stroke(width = 2.dp.toPx())
                )
            }

            drawCircle(
                color = pointColor,
                radius = if (isSelected) 7.dp.toPx() else 4.5.dp.toPx(),
                center = Offset(px, py)
            )
        }
    }
}

@Composable
fun PointDetailCard(
    point: DotPoint,
    onClose: () -> Unit,
    modifier: Modifier = Modifier
) {
    Card(
        modifier = modifier.fillMaxWidth(),
        shape = RoundedCornerShape(8.dp),
        colors = CardDefaults.cardColors(containerColor = SurfaceDark),
        border = CardDefaults.outlinedCardBorder()
    ) {
        Column(modifier = Modifier.padding(10.dp), verticalArrangement = Arrangement.spacedBy(6.dp)) {
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically
            ) {
                Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                    Surface(
                        shape = RoundedCornerShape(4.dp),
                        color = Color(point.group.colorHex).copy(alpha = 0.2f)
                    ) {
                        Text(
                            text = "Group ${point.group.id}",
                            modifier = Modifier.padding(horizontal = 6.dp, vertical = 2.dp),
                            style = MaterialTheme.typography.labelMedium,
                            color = Color(point.group.colorHex),
                            fontWeight = FontWeight.Bold
                        )
                    }
                    Text(
                        text = "ID: ${point.id}",
                        style = MaterialTheme.typography.bodySmall,
                        color = TextSecondary
                    )
                }

                TextButton(onClick = onClose, contentPadding = PaddingValues(0.dp)) {
                    Text("Close", fontSize = 11.sp, color = TextMuted)
                }
            }

            // Feature Vector Display
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween
            ) {
                FeatureType.entries.forEach { f ->
                    val value = point.features[f.key] ?: 0f
                    Column(horizontalAlignment = Alignment.CenterHorizontally) {
                        Text(f.displayName, fontSize = 9.sp, color = TextMuted)
                        Text(
                            text = when (f) {
                                FeatureType.IMPACT -> String.format("%.2f", value)
                                FeatureType.DURATION -> "${value.toInt()}ms"
                                FeatureType.PEAK_TIME -> "${value.toInt()}ms"
                                FeatureType.MEL_MEAN_DB -> String.format("%.1fdB", value)
                                FeatureType.IMPACT_FREQ_HZ -> "${value.toInt()}Hz"
                            },
                            fontFamily = FontFamily.Monospace,
                            fontSize = 10.sp,
                            color = TextPrimary,
                            fontWeight = FontWeight.SemiBold
                        )
                    }
                }
            }
        }
    }
}
