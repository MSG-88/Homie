package com.heyhomie.app.ui.screens

import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Text
import androidx.compose.runtime.*
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import androidx.hilt.navigation.compose.hiltViewModel
import com.heyhomie.app.ui.components.*
import com.heyhomie.app.ui.theme.*
import com.heyhomie.app.ui.viewmodel.PhoneStatsViewModel

@Composable
fun PhoneStatsScreen(viewModel: PhoneStatsViewModel = hiltViewModel()) {
    val profile by viewModel.profile.collectAsState()

    Column(
        modifier = Modifier
            .fillMaxSize()
            .background(RetroDark)
            .verticalScroll(rememberScrollState())
            .padding(16.dp),
        verticalArrangement = Arrangement.spacedBy(16.dp)
    ) {
        Text("DEVICE STATS", style = RetroTypography.headlineMedium)

        profile?.let { p ->
            RetroCard {
                Column {
                    Text("CAPABILITY SCORE", style = RetroTypography.titleMedium)
                    Spacer(Modifier.height(8.dp))
                    Text(
                        "${p.capabilityScore}/100",
                        style = RetroTypography.displayLarge,
                        color = when {
                            p.capabilityScore >= 70 -> RetroGreen
                            p.capabilityScore >= 50 -> RetroAmber
                            else -> RetroRed
                        }
                    )
                    Text(
                        "RECOMMENDED: ${p.recommendedModelSize}",
                        style = RetroTypography.labelMedium,
                        color = RetroCyan
                    )
                }
            }

            RetroCard {
                StatBar(
                    label = "\u2665 HP (BATTERY)",
                    value = p.batteryLevel.toFloat(),
                    color = when {
                        p.batteryLevel > 50 -> RetroGreen
                        p.batteryLevel > 20 -> RetroAmber
                        else -> RetroRed
                    },
                    suffix = "%"
                )
            }

            RetroCard {
                val usedPct = ((p.totalStorageMb - p.freeStorageMb).toFloat() / p.totalStorageMb * 100)
                StatBar(
                    label = "\u2605 XP (STORAGE)",
                    value = usedPct,
                    color = RetroCyan,
                    suffix = "% used"
                )
            }

            RetroCard {
                StatBar(
                    label = "\u25C6 MP (RAM)",
                    value = p.availableRamMb.toFloat(),
                    maxValue = p.totalRamMb.toFloat(),
                    color = RetroAmber,
                    suffix = " MB free"
                )
            }

            RetroCard {
                Column {
                    Text("CPU", style = RetroTypography.titleMedium)
                    Text("Cores: ${p.cpuCores}", style = RetroTypography.bodyMedium)
                    Text("Arch: ${p.cpuArch}", style = RetroTypography.bodyMedium)
                }
            }

            RetroCard {
                Column {
                    Text("GPU", style = RetroTypography.titleMedium)
                    Text("Renderer: ${p.gpuRenderer}", style = RetroTypography.bodyMedium)
                    Text(
                        "Vulkan: ${if (p.supportsVulkan) "YES" else "NO"}",
                        style = RetroTypography.bodyMedium,
                        color = if (p.supportsVulkan) RetroGreen else RetroRed
                    )
                }
            }

            if (p.isCharging) {
                Text("\u26A1 CHARGING", style = RetroTypography.labelMedium, color = RetroAmber)
            }
        } ?: Text("Scanning device...", style = RetroTypography.bodyMedium, color = RetroCyan)
    }
}
