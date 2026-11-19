package com.dsparena.c85.ui.components

import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Warning
import androidx.compose.material3.*
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.dsparena.c85.ui.model.GroupLabel
import com.dsparena.c85.ui.model.HonestyMetadata
import com.dsparena.c85.ui.theme.*

@Composable
fun HonestyCard(
    metadata: HonestyMetadata,
    reportedAccuracyOrLoo: Float? = null,
    modifier: Modifier = Modifier
) {
    Card(
        modifier = modifier
            .fillMaxWidth()
            .border(1.dp, OutlineDark.copy(alpha = 0.5f), RoundedCornerShape(12.dp)),
        shape = RoundedCornerShape(12.dp),
        colors = CardDefaults.cardColors(containerColor = SurfaceDark.copy(alpha = 0.95f))
    ) {
        Column(
            modifier = Modifier
                .padding(14.dp)
                .fillMaxWidth(),
            verticalArrangement = Arrangement.spacedBy(10.dp)
        ) {
            // Header Row: Honesty Title & Source Badge
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically
            ) {
                Text(
                    text = "Honesty & Provenance Layer",
                    style = MaterialTheme.typography.labelLarge,
                    color = TextSecondary,
                    fontWeight = FontWeight.SemiBold
                )
                Surface(
                    shape = RoundedCornerShape(6.dp),
                    color = SurfaceVariantDark,
                    border = CardDefaults.outlinedCardBorder()
                ) {
                    Text(
                        text = metadata.source.badge,
                        modifier = Modifier.padding(horizontal = 8.dp, vertical = 4.dp),
                        style = MaterialTheme.typography.labelSmall,
                        color = PrimaryNeon,
                        fontWeight = FontWeight.Bold
                    )
                }
            }

            // Group Sample Counts (n per group)
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(8.dp),
                verticalAlignment = Alignment.CenterVertically
            ) {
                Text(
                    text = "n per group:",
                    style = MaterialTheme.typography.bodySmall,
                    color = TextSecondary
                )
                GroupLabel.entries.forEach { group ->
                    val count = metadata.countsPerGroup[group] ?: 0
                    val groupColor = Color(group.colorHex)
                    Surface(
                        shape = RoundedCornerShape(4.dp),
                        color = groupColor.copy(alpha = 0.15f),
                        border = androidx.compose.foundation.BorderStroke(1.dp, groupColor.copy(alpha = 0.4f))
                    ) {
                        Text(
                            text = "${group.id}: $count",
                            modifier = Modifier.padding(horizontal = 6.dp, vertical = 2.dp),
                            style = MaterialTheme.typography.labelSmall,
                            color = groupColor,
                            fontWeight = FontWeight.Bold
                        )
                    }
                }
            }

            // Low n Warning
            if (metadata.hasLowNWarning) {
                Surface(
                    modifier = Modifier.fillMaxWidth(),
                    shape = RoundedCornerShape(6.dp),
                    color = VerdictAmber.copy(alpha = 0.15f),
                    border = androidx.compose.foundation.BorderStroke(1.dp, VerdictAmber.copy(alpha = 0.5f))
                ) {
                    Row(
                        modifier = Modifier.padding(8.dp),
                        verticalAlignment = Alignment.CenterVertically,
                        horizontalArrangement = Arrangement.spacedBy(6.dp)
                    ) {
                        Icon(
                            imageVector = Icons.Default.Warning,
                            contentDescription = "Warning",
                            tint = VerdictAmber,
                            modifier = Modifier.size(16.dp)
                        )
                        Text(
                            text = "⚠️ n နည်းလွန်း — TR→TE မကူးနိုင် (Small sample size: train-to-test transfer fragile)",
                            style = MaterialTheme.typography.bodySmall,
                            color = VerdictAmber,
                            fontSize = 11.sp
                        )
                    }
                }
            }

            // Session Badge & Inferred status
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically
            ) {
                Text(
                    text = "Session: [${metadata.sessionRange}]" + if (metadata.isInferredSession) " (ခန့်မှန်း)" else "",
                    style = MaterialTheme.typography.bodySmall,
                    color = TextSecondary
                )
            }

            // Cross Session Warning
            if (metadata.isCrossSession) {
                Surface(
                    modifier = Modifier.fillMaxWidth(),
                    shape = RoundedCornerShape(6.dp),
                    color = VerdictRed.copy(alpha = 0.15f),
                    border = androidx.compose.foundation.BorderStroke(1.dp, VerdictRed.copy(alpha = 0.5f))
                ) {
                    Text(
                        text = "⚠️ 15-min-apart sessions don't transfer (verified 2026-08-09 pitfall 20)",
                        modifier = Modifier.padding(8.dp),
                        style = MaterialTheme.typography.bodySmall,
                        color = VerdictRed,
                        fontSize = 11.sp
                    )
                }
            }

            // DSP Parameter Settings Log
            Surface(
                modifier = Modifier.fillMaxWidth(),
                shape = RoundedCornerShape(6.dp),
                color = DarkBackground.copy(alpha = 0.8f)
            ) {
                Text(
                    text = metadata.dspParamSummary,
                    modifier = Modifier.padding(horizontal = 8.dp, vertical = 6.dp),
                    fontFamily = FontFamily.Monospace,
                    fontSize = 10.sp,
                    color = TextSecondary
                )
            }

            // Honest Ceiling Banner
            Surface(
                modifier = Modifier.fillMaxWidth(),
                shape = RoundedCornerShape(6.dp),
                color = SurfaceVariantDark.copy(alpha = 0.6f)
            ) {
                Column(modifier = Modifier.padding(8.dp)) {
                    Text(
                        text = "🛡️ Honest Ceiling: 4-way ~71%, 1/2 ~86% max. A green 100% is a bug, not a result.",
                        style = MaterialTheme.typography.labelSmall,
                        color = TextMuted,
                        fontSize = 10.5.sp
                    )

                    if (reportedAccuracyOrLoo != null && reportedAccuracyOrLoo > 86f) {
                        Text(
                            text = "🚨 Ceiling violation detected: ${reportedAccuracyOrLoo.toInt()}% exceeds 86% physical upper bound!",
                            style = MaterialTheme.typography.labelSmall,
                            color = VerdictRed,
                            fontWeight = FontWeight.Bold,
                            modifier = Modifier.padding(top = 4.dp)
                        )
                    }
                }
            }
        }
    }
}
