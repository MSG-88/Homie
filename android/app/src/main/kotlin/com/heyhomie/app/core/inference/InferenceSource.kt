package com.heyhomie.app.core.inference

interface LocalInferenceBridge {
    val isAvailable: Boolean
    val modelName: String
    suspend fun generate(prompt: String, systemPrompt: String? = null): String
}

enum class InferenceSourceType { LOCAL, LAN, QUBRID }
