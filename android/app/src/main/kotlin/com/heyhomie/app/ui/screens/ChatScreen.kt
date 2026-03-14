package com.heyhomie.app.ui.screens

import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.lazy.rememberLazyListState
import androidx.compose.material3.Text
import androidx.compose.runtime.*
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import androidx.hilt.navigation.compose.hiltViewModel
import com.heyhomie.app.ui.components.*
import com.heyhomie.app.ui.theme.RetroAmber
import com.heyhomie.app.ui.theme.RetroCyan
import com.heyhomie.app.ui.theme.RetroDark
import com.heyhomie.app.ui.theme.RetroDarkCard
import com.heyhomie.app.ui.theme.RetroTypography
import com.heyhomie.app.ui.viewmodel.ChatViewModel

@Composable
fun ChatScreen(viewModel: ChatViewModel = hiltViewModel()) {
    val messages by viewModel.messages.collectAsState()
    val inputText by viewModel.inputText.collectAsState()
    val isGenerating by viewModel.isGenerating.collectAsState()
    val listState = rememberLazyListState()

    LaunchedEffect(messages.size) {
        if (messages.isNotEmpty()) {
            listState.animateScrollToItem(messages.lastIndex)
        }
    }

    Box(Modifier.fillMaxSize().background(RetroDark)) {
        Column(Modifier.fillMaxSize()) {
            // Fallback banner
            val fallbackBanner = viewModel.fallbackBanner
            if (fallbackBanner != null) {
                Text(
                    text = fallbackBanner,
                    style = RetroTypography.labelMedium,
                    color = RetroAmber,
                    modifier = Modifier
                        .fillMaxWidth()
                        .background(RetroDarkCard)
                        .padding(8.dp)
                )
            }

            // Inference source label
            Text(
                text = "SOURCE: ${viewModel.inferenceSource}",
                style = RetroTypography.labelMedium,
                color = RetroCyan,
                modifier = Modifier.padding(horizontal = 12.dp, vertical = 4.dp)
            )

            // Chat messages
            LazyColumn(
                state = listState,
                modifier = Modifier.weight(1f).padding(horizontal = 12.dp),
                verticalArrangement = Arrangement.spacedBy(8.dp),
                contentPadding = PaddingValues(vertical = 8.dp)
            ) {
                items(messages, key = { it.id }) { message ->
                    ChatBubble(message = message)
                }
            }

            // Input
            RetroTextField(
                value = inputText,
                onValueChange = { viewModel.updateInput(it) },
                onSend = { viewModel.sendMessage(inputText) },
                enabled = !isGenerating,
                modifier = Modifier.padding(8.dp)
            )
        }

        ScanlineOverlay(alpha = 0.03f)
    }
}
