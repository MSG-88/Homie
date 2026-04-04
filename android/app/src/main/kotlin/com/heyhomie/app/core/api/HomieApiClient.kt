package com.heyhomie.app.core.api

import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.channels.awaitClose
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.callbackFlow
import kotlinx.coroutines.withContext
import okhttp3.*
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.RequestBody.Companion.toRequestBody
import org.json.JSONObject
import java.util.concurrent.TimeUnit
import javax.inject.Inject
import javax.inject.Singleton

@Singleton
class HomieApiClient @Inject constructor() {
    private val client = OkHttpClient.Builder().connectTimeout(10, TimeUnit.SECONDS).readTimeout(60, TimeUnit.SECONDS).build()
    private var baseUrl: String = ""
    fun configure(serverUrl: String) { baseUrl = serverUrl.trimEnd('/') }
    val isConfigured: Boolean get() = baseUrl.isNotBlank()

    suspend fun healthCheck(): Boolean = withContext(Dispatchers.IO) {
        if (baseUrl.isBlank()) return@withContext false
        try { client.newCall(Request.Builder().url("$baseUrl/health").get().build()).execute().isSuccessful }
        catch (_: Exception) { false }
    }

    suspend fun sendMessage(message: String, conversationId: String): String = withContext(Dispatchers.IO) {
        require(baseUrl.isNotBlank()) { "Server URL not configured" }
        val body = JSONObject().apply { put("message", message); put("conversation_id", conversationId) }
        val request = Request.Builder().url("$baseUrl/api/chat").addHeader("Content-Type", "application/json")
            .post(body.toString().toRequestBody("application/json".toMediaType())).build()
        val response = client.newCall(request).execute()
        if (!response.isSuccessful) throw RuntimeException("Chat API error: ${response.code} ${response.message}")
        val json = JSONObject(response.body?.string() ?: "{}")
        json.optString("response", json.optString("message", ""))
    }

    fun streamMessages(conversationId: String): Flow<String> = callbackFlow {
        require(baseUrl.isNotBlank()) { "Server URL not configured" }
        val wsUrl = baseUrl.replace("http://", "ws://").replace("https://", "wss://")
        val ws = client.newWebSocket(Request.Builder().url("$wsUrl/ws").build(), object : WebSocketListener() {
            override fun onOpen(webSocket: WebSocket, response: Response) {
                webSocket.send(JSONObject().apply { put("type", "subscribe"); put("conversation_id", conversationId) }.toString())
            }
            override fun onMessage(webSocket: WebSocket, text: String) {
                try {
                    val json = JSONObject(text); val chunk = json.optString("chunk", json.optString("text", ""))
                    if (chunk.isNotEmpty()) trySend(chunk)
                    if (json.optBoolean("done", false)) close()
                } catch (_: Exception) { trySend(text) }
            }
            override fun onFailure(webSocket: WebSocket, t: Throwable, response: Response?) { close(t) }
            override fun onClosed(webSocket: WebSocket, code: Int, reason: String) { close() }
        })
        awaitClose { ws.close(1000, "Done") }
    }

    suspend fun getBriefing(): String = withContext(Dispatchers.IO) {
        require(baseUrl.isNotBlank()) { "Server URL not configured" }
        val response = client.newCall(Request.Builder().url("$baseUrl/api/briefing").get().build()).execute()
        if (!response.isSuccessful) throw RuntimeException("Briefing API error: ${response.code}")
        val json = JSONObject(response.body?.string() ?: "{}")
        json.optString("briefing", json.optString("message", "No briefing available."))
    }
}
