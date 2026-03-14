package com.heyhomie.app.ui.viewmodel

import androidx.lifecycle.ViewModel
import com.heyhomie.app.network.*
import dagger.hilt.android.lifecycle.HiltViewModel
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import javax.inject.Inject

@HiltViewModel
class NetworkViewModel @Inject constructor(
    private val discovery: LanDiscovery,
    private val syncClient: SyncClient
) : ViewModel() {
    val peers: StateFlow<List<PeerDevice>> = discovery.peers
    val connectionState: StateFlow<ConnectionState> = syncClient.connectionState

    private val _pairingCode = MutableStateFlow("")
    val pairingCode: StateFlow<String> = _pairingCode

    init {
        discovery.startDiscovery()
    }

    fun updatePairingCode(code: String) {
        _pairingCode.value = code.filter { it.isDigit() }.take(6)
    }

    fun connectToPeer(peer: PeerDevice) {
        syncClient.connect(peer)
    }

    fun disconnect() {
        syncClient.disconnect()
    }

    override fun onCleared() {
        discovery.stopDiscovery()
        syncClient.disconnect()
    }
}
