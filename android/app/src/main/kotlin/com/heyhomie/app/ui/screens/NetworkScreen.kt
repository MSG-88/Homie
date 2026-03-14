package com.heyhomie.app.ui.screens

import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.material3.Text
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import androidx.hilt.navigation.compose.hiltViewModel
import com.heyhomie.app.network.ConnectionState
import com.heyhomie.app.ui.components.*
import com.heyhomie.app.ui.theme.*
import com.heyhomie.app.ui.viewmodel.NetworkViewModel

@Composable
fun NetworkScreen(viewModel: NetworkViewModel = hiltViewModel()) {
    val peers by viewModel.peers.collectAsState()
    val connectionState by viewModel.connectionState.collectAsState()

    Column(
        modifier = Modifier
            .fillMaxSize()
            .background(RetroDark)
            .padding(16.dp),
        verticalArrangement = Arrangement.spacedBy(16.dp)
    ) {
        Text("LAN SYNC", style = RetroTypography.headlineMedium)

        RetroCard {
            Row(verticalAlignment = Alignment.CenterVertically) {
                val (statusText, statusColor) = when (connectionState) {
                    ConnectionState.CONNECTED -> "\u25CF CONNECTED" to RetroGreen
                    ConnectionState.CONNECTING -> "\u25CC CONNECTING..." to RetroAmber
                    ConnectionState.DISCONNECTED -> "\u25CB OFFLINE" to RetroRed
                }
                Text(statusText, style = RetroTypography.titleMedium, color = statusColor)
            }
        }

        if (connectionState == ConnectionState.CONNECTED) {
            RetroCard {
                Text(
                    "\u2605 PLAYER 2 HAS JOINED \u2605",
                    style = RetroTypography.titleMedium,
                    color = RetroAmber
                )
            }
        }

        Text("NEARBY DEVICES", style = RetroTypography.titleMedium)

        if (peers.isEmpty()) {
            RetroCard {
                Text(
                    "Scanning LAN...\nMake sure Homie is running on your desktop.",
                    style = RetroTypography.bodyMedium,
                    color = RetroCyan
                )
            }
        } else {
            LazyColumn(verticalArrangement = Arrangement.spacedBy(8.dp)) {
                items(peers) { peer ->
                    RetroCard(
                        modifier = Modifier.clickable { viewModel.connectToPeer(peer) }
                    ) {
                        Column {
                            Text(peer.name, style = RetroTypography.titleMedium, color = RetroGreen)
                            Text(
                                "${peer.host}:${peer.port}",
                                style = RetroTypography.bodyMedium,
                                color = RetroGray
                            )
                            Text("[CONNECT]", style = RetroTypography.labelMedium, color = RetroAmber)
                        }
                    }
                }
            }
        }

        if (connectionState == ConnectionState.CONNECTED) {
            Text(
                "[DISCONNECT]",
                style = RetroTypography.labelMedium,
                color = RetroRed,
                modifier = Modifier
                    .clickable { viewModel.disconnect() }
                    .pixelBorder(RetroRed, width = 1f)
                    .padding(8.dp)
            )
        }
    }
}
