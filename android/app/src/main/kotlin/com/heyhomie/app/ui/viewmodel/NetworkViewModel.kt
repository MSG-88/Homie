package com.heyhomie.app.ui.viewmodel

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.heyhomie.app.core.api.HomieApiClient
import com.heyhomie.app.core.config.SettingsStore
import com.heyhomie.app.network.ConnectionState
import com.heyhomie.app.network.LanDiscovery
import com.heyhomie.app.network.PeerDevice
import com.heyhomie.app.network.SyncClient
import dagger.hilt.android.lifecycle.HiltViewModel
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.launch
import javax.inject.Inject

@HiltViewModel
class NetworkViewModel @Inject constructor(
    private val discovery: LanDiscovery, private val syncClient: SyncClient,
    private val apiClient: HomieApiClient, private val settingsStore: SettingsStore
) : ViewModel() {
    val peers: StateFlow<List<PeerDevice>> = discovery.peers
    val connectionState: StateFlow<ConnectionState> = syncClient.connectionState
    private val _manualIp = MutableStateFlow("")
    val manualIp: StateFlow<String> = _manualIp
    private val _manualPort = MutableStateFlow("3141")
    val manualPort: StateFlow<String> = _manualPort
    private val _connectionResult = MutableStateFlow<String?>(null)
    val connectionResult: StateFlow<String?> = _connectionResult
    init {
        discovery.startDiscovery()
        viewModelScope.launch {
            val h = settingsStore.lastConnectedHost.first()
            val p = settingsStore.lastConnectedPort.first()
            if (h.isNotBlank()) { _manualIp.value = h; _manualPort.value = p.toString() }
        }
    }
    fun updateManualIp(ip: String) { _manualIp.value = ip }
    fun updateManualPort(port: String) { _manualPort.value = port.filter { it.isDigit() } }
    fun connectToPeer(peer: PeerDevice) {
        val u = "http://" + peer.host + ":" + peer.port
        apiClient.configure(u); syncClient.connect(peer)
        viewModelScope.launch { settingsStore.setServerUrl(u); settingsStore.setLastConnected(peer.host, peer.port) }
    }
    fun connectManual() {
        val ip = _manualIp.value.trim(); val port = _manualPort.value.toIntOrNull() ?: 3141
        if (ip.isBlank()) { _connectionResult.value = "Enter an IP address"; return }
        val u = "http://" + ip + ":" + port
        viewModelScope.launch {
            apiClient.configure(u)
            if (apiClient.healthCheck()) {
                settingsStore.setServerUrl(u); settingsStore.setLastConnected(ip, port)
                syncClient.connect(PeerDevice("Homie", ip, port, "manual-" + ip))
                _connectionResult.value = "Connected to " + ip + ":" + port
            } else _connectionResult.value = "Could not reach Homie at " + ip + ":" + port
        }
    }
    fun disconnect() { syncClient.disconnect(); _connectionResult.value = null }
    override fun onCleared() { discovery.stopDiscovery(); syncClient.disconnect() }
}
