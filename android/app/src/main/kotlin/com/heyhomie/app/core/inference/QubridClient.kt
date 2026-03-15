package com.heyhomie.app.core.inference

import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import okhttp3.*
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.RequestBody.Companion.toRequestBody
import org.json.JSONArray
import org.json.JSONObject
import java.util.concurrent.TimeUnit

class QubridClient(
    private val apiKey: String,
    val baseUrl: String = "https://platform.qubrid.com/v1",
    val model: String = "Qwen/Qwen3.5-Flash",
    private val timeout: Long = 60_000
) {
    private val client = OkHttpClient.Builder()
        .connectTimeout(timeout, TimeUnit.MILLISECONDS)
        .readTimeout(timeout, TimeUnit.MILLISECONDS)
        .build()

    val isAvailable: Boolean get() = apiKey.isNotBlank()

    suspend fun generate(
        prompt: String,
        systemPrompt: String? = null,
        modelHint: String? = null
    ): String =
        withContext(Dispatchers.IO) {
            val messages = JSONArray().apply {
                systemPrompt?.let {
                    put(JSONObject().put("role", "system").put("content", it))
                }
                put(JSONObject().put("role", "user").put("content", prompt))
            }
            val body = JSONObject().apply {
                put("model", modelHint ?: model)
                put("messages", messages)
                put("max_tokens", 2048)
            }
            val request = Request.Builder()
                .url("$baseUrl/chat/completions")
                .addHeader("Authorization", "Bearer $apiKey")
                .addHeader("Content-Type", "application/json")
                .post(body.toString().toRequestBody("application/json".toMediaType()))
                .build()

            val response = client.newCall(request).execute()
            if (!response.isSuccessful) {
                throw RuntimeException("Qubrid API error: ${response.code} ${response.message}")
            }
            val json = JSONObject(response.body?.string() ?: "")
            json.getJSONArray("choices")
                .getJSONObject(0)
                .getJSONObject("message")
                .getString("content")
        }
}
