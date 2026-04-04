package com.heyhomie.app.ui.screens

import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.text.BasicTextField
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Text
import androidx.compose.runtime.*
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.SolidColor
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
    val darkTheme by viewModel.darkTheme.collectAsState()
    val notificationsEnabled by viewModel.notificationsEnabled.collectAsState()
    val briefingEnabled by viewModel.briefingEnabled.collectAsState()
    val serverUrlInput by viewModel.serverUrlInput.collectAsState()
    val serverStatus by viewModel.serverStatus.collectAsState()
    Column(Modifier.fillMaxSize().background(RetroDark).verticalScroll(rememberScrollState()).padding(16.dp), verticalArrangement = Arrangement.spacedBy(12.dp)) {
        Text("CONFIG", style = RetroTypography.headlineMedium)
        Text("SERVER", style = RetroTypography.titleMedium)
        RetroCard { Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
            Text("Homie Server URL", style = RetroTypography.labelMedium, color = RetroCyan)
            Row(Modifier.fillMaxWidth().pixelBorder(RetroGreen, width = 1f).background(RetroDark).padding(8.dp)) {
                Text("> ", style = RetroTypography.bodyMedium, color = RetroGreen)
                BasicTextField(value = serverUrlInput, onValueChange = { viewModel.updateServerUrlInput(it) }, textStyle = RetroTypography.bodyMedium.copy(color = RetroWhite), cursorBrush = SolidColor(RetroGreen), modifier = Modifier.weight(1f), singleLine = true)
            }
            Text("e.g. http://192.168.1.100:3141", style = RetroTypography.labelMedium, color = RetroGray)
            Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                Text("[SAVE]", style = RetroTypography.labelMedium, color = RetroAmber, modifier = Modifier.clickable { viewModel.saveServerUrl() }.pixelBorder(RetroAmber, width = 1f).padding(4.dp))
                Text("[TEST]", style = RetroTypography.labelMedium, color = RetroCyan, modifier = Modifier.clickable { viewModel.testConnection() }.pixelBorder(RetroCyan, width = 1f).padding(4.dp))
            }
            serverStatus?.let { Text("STATUS: " + it, style = RetroTypography.labelMedium, color = if (it.contains("OK") || it.contains("Connected")) RetroGreen else RetroRed) }
        } }
        Spacer(Modifier.height(4.dp))
        Text("THEME", style = RetroTypography.titleMedium)
        RetroToggle("Dark mode", darkTheme) { viewModel.toggleDarkTheme() }
        Spacer(Modifier.height(4.dp))
        Text("DISPLAY", style = RetroTypography.titleMedium)
        RetroToggle("Scanline overlay", scanlines) { viewModel.toggleScanlines() }
        RetroToggle("High contrast mode", highContrast) { viewModel.toggleHighContrast() }
        RetroToggle("8-bit sound FX", soundEffects) { viewModel.toggleSoundEffects() }
        Spacer(Modifier.height(4.dp))
        Text("NOTIFICATIONS", style = RetroTypography.titleMedium)
        RetroToggle("Push notifications", notificationsEnabled) { viewModel.toggleNotifications() }
        RetroToggle("Daily briefing", briefingEnabled) { viewModel.toggleBriefing() }
        Spacer(Modifier.height(4.dp))
        Text("SYNC SCOPE", style = RetroTypography.titleMedium)
        listOf("all" to "ALL MEMORY", "conversations" to "CONVERSATIONS ONLY", "manual" to "MANUAL").forEach { (value, label) ->
            val selected = syncScope == value
            Text((if (selected) "\u25BA " else "  ") + label, style = RetroTypography.bodyMedium, color = if (selected) RetroGreen else RetroGray, modifier = Modifier.clickable { viewModel.setSyncScope(value) }.padding(vertical = 4.dp))
        }
        Spacer(Modifier.height(8.dp))
        Text("ABOUT", style = RetroTypography.titleMedium)
        RetroCard { Column {
            Text("HOMIE AI v0.3.0", style = RetroTypography.bodyMedium, color = RetroGreen)
            Text("Desktop companion app", style = RetroTypography.bodyMedium, color = RetroGray)
            Text("heyhomie.ai", style = RetroTypography.bodyMedium, color = RetroCyan)
        } }
    }
}

@Composable
private fun RetroToggle(label: String, enabled: Boolean, onToggle: () -> Unit) {
    Row(Modifier.fillMaxWidth().clickable(onClick = onToggle).pixelBorder(if (enabled) RetroGreen else RetroGray, width = 1f).padding(12.dp), horizontalArrangement = Arrangement.SpaceBetween) {
        Text(label, style = RetroTypography.bodyMedium)
        Text(if (enabled) "[ON]" else "[OFF]", style = RetroTypography.labelMedium, color = if (enabled) RetroGreen else RetroRed)
    }
}
