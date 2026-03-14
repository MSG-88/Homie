package com.heyhomie.app.ui.viewmodel

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.heyhomie.app.core.inference.InferenceRouter
import com.heyhomie.app.core.model.ChatMessage
import com.heyhomie.app.core.model.MessageRole
import dagger.hilt.android.lifecycle.HiltViewModel
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch
import javax.inject.Inject

@HiltViewModel
class ChatViewModel @Inject constructor(
    private val inferenceRouter: InferenceRouter
) : ViewModel() {
    private val _messages = MutableStateFlow<List<ChatMessage>>(emptyList())
    val messages: StateFlow<List<ChatMessage>> = _messages.asStateFlow()

    private val _inputText = MutableStateFlow("")
    val inputText: StateFlow<String> = _inputText.asStateFlow()

    private val _isGenerating = MutableStateFlow(false)
    val isGenerating: StateFlow<Boolean> = _isGenerating.asStateFlow()

    val fallbackBanner: String? get() = inferenceRouter.fallbackBanner
    val inferenceSource: String get() = inferenceRouter.activeSourceName

    fun updateInput(text: String) { _inputText.value = text }

    fun sendMessage(text: String) {
        if (text.isBlank() || _isGenerating.value) return
        val userMsg = ChatMessage(role = MessageRole.USER, text = text.trim())
        _messages.value = _messages.value + userMsg
        _inputText.value = ""

        viewModelScope.launch {
            _isGenerating.value = true
            try {
                val response = inferenceRouter.generate(
                    prompt = text.trim(),
                    systemPrompt = "You are Homie, a friendly local-first AI assistant. Be helpful, concise, and warm."
                )
                val assistantMsg = ChatMessage(
                    role = MessageRole.ASSISTANT,
                    text = response,
                    isStreaming = true
                )
                _messages.value = _messages.value + assistantMsg
            } catch (e: Exception) {
                val errorMsg = ChatMessage(
                    role = MessageRole.SYSTEM,
                    text = "ERROR: ${e.message ?: "Inference failed"}"
                )
                _messages.value = _messages.value + errorMsg
            } finally {
                _isGenerating.value = false
            }
        }
    }
}
