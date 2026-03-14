package com.heyhomie.app.ui.viewmodel

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.heyhomie.app.core.config.SettingsStore
import dagger.hilt.android.lifecycle.HiltViewModel
import kotlinx.coroutines.flow.SharingStarted
import kotlinx.coroutines.flow.stateIn
import kotlinx.coroutines.launch
import javax.inject.Inject

@HiltViewModel
class SettingsViewModel @Inject constructor(
    private val settings: SettingsStore
) : ViewModel() {
    val scanlines = settings.scanlines.stateIn(viewModelScope, SharingStarted.Eagerly, true)
    val highContrast = settings.highContrast.stateIn(viewModelScope, SharingStarted.Eagerly, false)
    val soundEffects = settings.soundEffects.stateIn(viewModelScope, SharingStarted.Eagerly, true)
    val syncScope = settings.syncScope.stateIn(viewModelScope, SharingStarted.Eagerly, "all")

    fun toggleScanlines() = viewModelScope.launch { settings.setScanlines(!scanlines.value) }
    fun toggleHighContrast() = viewModelScope.launch { settings.setHighContrast(!highContrast.value) }
    fun toggleSoundEffects() = viewModelScope.launch { settings.setSoundEffects(!soundEffects.value) }
    fun setSyncScope(scope: String) = viewModelScope.launch { settings.setSyncScope(scope) }
}
