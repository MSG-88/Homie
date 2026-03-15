package com.heyhomie.app.core.inference

class InferenceRouter(
    private val localBridge: LocalInferenceBridge? = null,
    private val qubridClient: QubridClient? = null
) {
    companion object {
        private const val FALLBACK_BANNER =
            "No local model found! Using Homie's intelligence until local model is setup!"
    }

    val activeSourceName: String
        get() = when {
            localBridge?.isAvailable == true -> "Local (${localBridge.modelName})"
            qubridClient?.isAvailable == true -> "Homie Intelligence (Cloud)"
            else -> "None"
        }

    val activeSourceType: InferenceSourceType?
        get() = when {
            localBridge?.isAvailable == true -> InferenceSourceType.LOCAL
            qubridClient?.isAvailable == true -> InferenceSourceType.QUBRID
            else -> null
        }

    val fallbackBanner: String?
        get() = when {
            localBridge?.isAvailable == true -> null
            qubridClient?.isAvailable == true -> FALLBACK_BANNER
            else -> null
        }

    suspend fun generate(
        prompt: String,
        systemPrompt: String? = null,
        model: String? = null,
        preferredLocation: String? = null
    ): String {
        if (preferredLocation != "cloud" && localBridge?.isAvailable == true) {
            return try {
                localBridge.generate(prompt, systemPrompt)
            } catch (e: Exception) {
                fallbackGenerate(prompt, systemPrompt, model)
            }
        }
        return fallbackGenerate(prompt, systemPrompt, model)
    }

    private suspend fun fallbackGenerate(
        prompt: String,
        systemPrompt: String?,
        model: String? = null
    ): String {
        if (qubridClient?.isAvailable == true) {
            return qubridClient.generate(prompt, systemPrompt, model)
        }
        throw IllegalStateException("No inference source available")
    }
}
