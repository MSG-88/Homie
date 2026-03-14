package com.heyhomie.app.ui.screens

import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
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
import com.heyhomie.app.ui.viewmodel.SettingsViewModel

@Composable
fun SettingsScreen(viewModel: SettingsViewModel = hiltViewModel()) {
    val scanlines by viewModel.scanlines.collectAsState()
    val highContrast by viewModel.highContrast.collectAsState()
    val soundEffects by viewModel.soundEffects.collectAsState()
    val syncScope by viewModel.syncScope.collectAsState()

    Column(
        modifier = Modifier
            .fillMaxSize()
            .background(RetroDark)
            .verticalScroll(rememberScrollState())
            .padding(16.dp),
        verticalArrangement = Arrangement.spacedBy(12.dp)
    ) {
        Text("CONFIG", style = RetroTypography.headlineMedium)

        Text("DISPLAY", style = RetroTypography.titleMedium)
        RetroToggle("Scanline overlay", scanlines) { viewModel.toggleScanlines() }
        RetroToggle("High contrast mode", highContrast) { viewModel.toggleHighContrast() }
        RetroToggle("8-bit sound FX", soundEffects) { viewModel.toggleSoundEffects() }

        Spacer(Modifier.height(8.dp))

        Text("SYNC SCOPE", style = RetroTypography.titleMedium)
        listOf("all" to "ALL MEMORY", "conversations" to "CONVERSATIONS ONLY", "manual" to "MANUAL")
            .forEach { (value, label) ->
                val selected = syncScope == value
                Text(
                    text = "${if (selected) "\u25BA" else " "} $label",
                    style = RetroTypography.bodyMedium,
                    color = if (selected) RetroGreen else RetroGray,
                    modifier = Modifier
                        .clickable { viewModel.setSyncScope(value) }
                        .padding(vertical = 4.dp)
                )
            }

        Spacer(Modifier.height(8.dp))

        Text("ABOUT", style = RetroTypography.titleMedium)
        RetroCard {
            Column {
                Text("HOMIE AI v0.1.0", style = RetroTypography.bodyMedium, color = RetroGreen)
                Text("Local-first AI assistant", style = RetroTypography.bodyMedium, color = RetroGray)
                Text("heyhomie.ai", style = RetroTypography.bodyMedium, color = RetroCyan)
            }
        }
    }
}

@Composable
private fun RetroToggle(label: String, enabled: Boolean, onToggle: () -> Unit) {
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .clickable(onClick = onToggle)
            .pixelBorder(if (enabled) RetroGreen else RetroGray, width = 1f)
            .padding(12.dp),
        horizontalArrangement = Arrangement.SpaceBetween
    ) {
        Text(label, style = RetroTypography.bodyMedium)
        Text(
            if (enabled) "[ON]" else "[OFF]",
            style = RetroTypography.labelMedium,
            color = if (enabled) RetroGreen else RetroRed
        )
    }
}
