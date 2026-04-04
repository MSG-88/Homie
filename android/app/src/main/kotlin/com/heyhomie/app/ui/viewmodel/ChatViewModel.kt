package com.heyhomie.app.ui.viewmodel

import android.app.Application
import android.content.Intent
import android.os.Bundle
import android.speech.RecognitionListener
import android.speech.RecognizerIntent
import android.speech.SpeechRecognizer
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.heyhomie.app.core.api.HomieApiClient
import com.heyhomie.app.core.config.SettingsStore
import com.heyhomie.app.core.inference.InferenceRouter
import com.heyhomie.app.core.model.ChatMessage
import com.heyhomie.app.core.model.MessageRole
import com.heyhomie.app.core.repository.ChatRepository
import dagger.hilt.android.lifecycle.HiltViewModel
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.launch
import java.util.UUID
import javax.inject.Inject

@HiltViewModel
class ChatViewModel @Inject constructor(
    private val inferenceRouter: InferenceRouter,
    private val apiClient: HomieApiClient,
    private val chatRepository: ChatRepository,
    private val settingsStore: SettingsStore,
    private val application: Application
) : ViewModel() {
    private val _messages = MutableStateFlow<List<ChatMessage>>(emptyList())
    val messages: StateFlow<List<ChatMessage>> = _messages.asStateFlow()
    private val _inputText = MutableStateFlow("")
    val inputText: StateFlow<String> = _inputText.asStateFlow()
    private val _isGenerating = MutableStateFlow(false)
    val isGenerating: StateFlow<Boolean> = _isGenerating.asStateFlow()
    private val _isListening = MutableStateFlow(false)
    val isListening: StateFlow<Boolean> = _isListening.asStateFlow()
    private val _streamingText = MutableStateFlow("")
    val streamingText: StateFlow<String> = _streamingText.asStateFlow()
    private var conversationId: String = UUID.randomUUID().toString()
    private var speechRecognizer: SpeechRecognizer? = null
    val fallbackBanner: String?
        get() = if (apiClient.isConfigured) null else inferenceRouter.fallbackBanner ?: "Connect to a Homie instance to chat"
    val inferenceSource: String
        get() = if (apiClient.isConfigured) "Homie (LAN)" else inferenceRouter.activeSourceName
    init {
        viewModelScope.launch { val u = settingsStore.serverUrl.first(); if (u.isNotBlank()) apiClient.configure(u) }
        viewModelScope.launch { chatRepository.getMessages(conversationId).collect { if (it.isNotEmpty()) _messages.value = it } }
    }
    fun updateInput(text: String) { _inputText.value = text }
    fun sendMessage(text: String) {
        if (text.isBlank() || _isGenerating.value) return
        _messages.value = _messages.value + ChatMessage(role = MessageRole.USER, text = text.trim()); _inputText.value = ""
        viewModelScope.launch {
            _isGenerating.value = true
            try {
                val r = if (apiClient.isConfigured) {
                    chatRepository.saveMessage(conversationId, "user", text.trim())
                    val x = apiClient.sendMessage(text.trim(), conversationId)
                    chatRepository.saveMessage(conversationId, "assistant", x); x
                } else inferenceRouter.generate(prompt = text.trim(), systemPrompt = "You are Homie, a friendly local-first AI assistant.")
                _streamingText.value = ""; _messages.value = _messages.value + ChatMessage(role = MessageRole.ASSISTANT, text = r, isStreaming = true)
            } catch (e: Exception) { _streamingText.value = ""; _messages.value = _messages.value + ChatMessage(role = MessageRole.SYSTEM, text = "ERROR: ${e.message}") }
            finally { _isGenerating.value = false }
        }
    }
    fun startVoiceInput() {
        if (!SpeechRecognizer.isRecognitionAvailable(application)) return
        speechRecognizer?.destroy()
        speechRecognizer = SpeechRecognizer.createSpeechRecognizer(application).apply {
            setRecognitionListener(object : RecognitionListener {
                override fun onReadyForSpeech(p: Bundle?) { _isListening.value = true }
                override fun onBeginningOfSpeech() {}
                override fun onRmsChanged(r: Float) {}
                override fun onBufferReceived(b: ByteArray?) {}
                override fun onEndOfSpeech() { _isListening.value = false }
                override fun onError(e: Int) { _isListening.value = false }
                override fun onResults(r: Bundle?) { _isListening.value = false; r?.getStringArrayList(SpeechRecognizer.RESULTS_RECOGNITION)?.firstOrNull()?.let { _inputText.value = it } }
                override fun onPartialResults(p: Bundle?) { p?.getStringArrayList(SpeechRecognizer.RESULTS_RECOGNITION)?.firstOrNull()?.let { _inputText.value = it } }
                override fun onEvent(t: Int, p: Bundle?) {}
            })
        }
        speechRecognizer?.startListening(Intent(RecognizerIntent.ACTION_RECOGNIZE_SPEECH).apply {
            putExtra(RecognizerIntent.EXTRA_LANGUAGE_MODEL, RecognizerIntent.LANGUAGE_MODEL_FREE_FORM)
            putExtra(RecognizerIntent.EXTRA_PARTIAL_RESULTS, true)
            putExtra(RecognizerIntent.EXTRA_MAX_RESULTS, 1)
        })
    }
    fun stopVoiceInput() { speechRecognizer?.stopListening(); _isListening.value = false }
    fun newConversation() { conversationId = UUID.randomUUID().toString(); _messages.value = emptyList() }
    override fun onCleared() { speechRecognizer?.destroy(); speechRecognizer = null }
}
