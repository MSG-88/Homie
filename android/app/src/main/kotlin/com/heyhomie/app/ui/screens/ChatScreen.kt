package com.heyhomie.app.ui.screens

import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.lazy.rememberLazyListState
import androidx.compose.material3.Text
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import androidx.hilt.navigation.compose.hiltViewModel
import com.heyhomie.app.ui.components.*
import com.heyhomie.app.ui.theme.*
import com.heyhomie.app.ui.viewmodel.ChatViewModel

@Composable
fun ChatScreen(viewModel: ChatViewModel = hiltViewModel()) {
    val messages by viewModel.messages.collectAsState()
    val inputText by viewModel.inputText.collectAsState()
    val isGenerating by viewModel.isGenerating.collectAsState()
    val isListening by viewModel.isListening.collectAsState()
    val streamingText by viewModel.streamingText.collectAsState()
    val listState = rememberLazyListState()
    LaunchedEffect(messages.size) { if (messages.isNotEmpty()) listState.animateScrollToItem(messages.lastIndex) }
    Box(Modifier.fillMaxSize().background(RetroDark)) {
        Column(Modifier.fillMaxSize()) {
            val fallbackBanner = viewModel.fallbackBanner
            if (fallbackBanner != null) { Text(fallbackBanner, style = RetroTypography.labelMedium, color = RetroAmber, modifier = Modifier.fillMaxWidth().background(RetroDarkCard).padding(8.dp)) }
            Row(Modifier.fillMaxWidth().padding(horizontal = 12.dp, vertical = 4.dp), horizontalArrangement = Arrangement.SpaceBetween, verticalAlignment = Alignment.CenterVertically) {
                Text("SOURCE: " + viewModel.inferenceSource, style = RetroTypography.labelMedium, color = RetroCyan)
                Text("[NEW]", style = RetroTypography.labelMedium, color = RetroAmber, modifier = Modifier.clickable { viewModel.newConversation() }.pixelBorder(RetroAmber, width = 1f).padding(4.dp))
            }
            LazyColumn(state = listState, modifier = Modifier.weight(1f).padding(horizontal = 12.dp), verticalArrangement = Arrangement.spacedBy(8.dp), contentPadding = PaddingValues(vertical = 8.dp)) {
                items(messages, key = { it.id }) { ChatBubble(message = it) }
                if (streamingText.isNotBlank()) { item("streaming") { Text("HOMIE> " + streamingText + "\u2588", style = RetroTypography.bodyMedium, color = RetroGreen, modifier = Modifier.widthIn(max = 300.dp).crtGlow(RetroGreen, 8f).pixelBorder(RetroGreen, width = 1f).background(RetroDarkCard).padding(10.dp)) } }
                if (isGenerating && streamingText.isBlank()) { item("generating") { GeneratingIndicator() } }
            }
            Row(Modifier.fillMaxWidth().padding(8.dp), verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(4.dp)) {
                Text(if (isListening) "[...]" else "[MIC]", style = RetroTypography.labelMedium, color = if (isListening) RetroRed else RetroGreen,
                    modifier = Modifier.clickable { if (isListening) viewModel.stopVoiceInput() else viewModel.startVoiceInput() }.pixelBorder(if (isListening) RetroRed else RetroGreen, width = 1f).padding(8.dp))
                RetroTextField(value = inputText, onValueChange = { viewModel.updateInput(it) }, onSend = { viewModel.sendMessage(inputText) }, enabled = !isGenerating, modifier = Modifier.weight(1f))
            }
        }
        ScanlineOverlay(alpha = 0.03f)
    }
}

@Composable
private fun GeneratingIndicator() {
    var dots by remember { mutableIntStateOf(0) }
    LaunchedEffect(Unit) { while (true) { kotlinx.coroutines.delay(400); dots = (dots + 1) % 4 } }
    Box(Modifier.fillMaxWidth(), contentAlignment = Alignment.CenterStart) {
        Text("HOMIE> thinking" + ".".repeat(dots), style = RetroTypography.bodyMedium, color = RetroGreen, modifier = Modifier.pixelBorder(RetroGreen, width = 1f).background(RetroDarkCard).padding(10.dp))
    }
}
