package com.heyhomie.app.ui.viewmodel

import androidx.lifecycle.ViewModel
import com.heyhomie.app.phone.DeviceProfile
import com.heyhomie.app.phone.DeviceProfiler
import dagger.hilt.android.lifecycle.HiltViewModel
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import javax.inject.Inject

@HiltViewModel
class PhoneStatsViewModel @Inject constructor(
    private val profiler: DeviceProfiler
) : ViewModel() {
    private val _profile = MutableStateFlow<DeviceProfile?>(null)
    val profile: StateFlow<DeviceProfile?> = _profile

    init { refresh() }

    fun refresh() {
        _profile.value = profiler.profile()
    }
}
