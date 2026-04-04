package com.heyhomie.app.ui.viewmodel

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.heyhomie.app.core.api.HomieApiClient
import com.heyhomie.app.core.config.SettingsStore
import dagger.hilt.android.lifecycle.HiltViewModel
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.SharingStarted
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.stateIn
import kotlinx.coroutines.launch
import javax.inject.Inject

@HiltViewModel
class SettingsViewModel @Inject constructor(private val settings: SettingsStore, private val apiClient: HomieApiClient) : ViewModel() {
    val scanlines = settings.scanlines.stateIn(viewModelScope, SharingStarted.Eagerly, true)
    val highContrast = settings.highContrast.stateIn(viewModelScope, SharingStarted.Eagerly, false)
    val soundEffects = settings.soundEffects.stateIn(viewModelScope, SharingStarted.Eagerly, true)
    val syncScope = settings.syncScope.stateIn(viewModelScope, SharingStarted.Eagerly, "all")
    val serverUrl = settings.serverUrl.stateIn(viewModelScope, SharingStarted.Eagerly, "")
    val darkTheme = settings.darkTheme.stateIn(viewModelScope, SharingStarted.Eagerly, true)
    val notificationsEnabled = settings.notificationsEnabled.stateIn(viewModelScope, SharingStarted.Eagerly, true)
    val briefingEnabled = settings.briefingEnabled.stateIn(viewModelScope, SharingStarted.Eagerly, true)
    private val _serverUrlInput = MutableStateFlow("")
    val serverUrlInput: StateFlow<String> = _serverUrlInput
    private val _serverStatus = MutableStateFlow<String?>(null)
    val serverStatus: StateFlow<String?> = _serverStatus
    init { viewModelScope.launch { settings.serverUrl.collect { _serverUrlInput.value = it } } }
    fun updateServerUrlInput(url: String) { _serverUrlInput.value = url }
    fun saveServerUrl() { val u = _serverUrlInput.value.trim(); viewModelScope.launch { settings.setServerUrl(u); if (u.isNotBlank()) { apiClient.configure(u); _serverStatus.value = if (apiClient.healthCheck()) "Connected" else "Unreachable" } else _serverStatus.value = null } }
    fun testConnection() { viewModelScope.launch { val u = _serverUrlInput.value.trim(); if (u.isBlank()) { _serverStatus.value = "Enter a URL first"; return@launch }; apiClient.configure(u); _serverStatus.value = if (apiClient.healthCheck()) "OK - Connected" else "FAIL - Unreachable" } }
    fun toggleScanlines() = viewModelScope.launch { settings.setScanlines(!scanlines.value) }
    fun toggleHighContrast() = viewModelScope.launch { settings.setHighContrast(!highContrast.value) }
    fun toggleSoundEffects() = viewModelScope.launch { settings.setSoundEffects(!soundEffects.value) }
    fun toggleDarkTheme() = viewModelScope.launch { settings.setDarkTheme(!darkTheme.value) }
    fun toggleNotifications() = viewModelScope.launch { settings.setNotificationsEnabled(!notificationsEnabled.value) }
    fun toggleBriefing() = viewModelScope.launch { settings.setBriefingEnabled(!briefingEnabled.value) }
    fun setSyncScope(scope: String) = viewModelScope.launch { settings.setSyncScope(scope) }
}
