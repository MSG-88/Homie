package com.heyhomie.app.core.repository

import com.heyhomie.app.core.api.HomieApiClient
import com.heyhomie.app.core.data.dao.MessageDao
import com.heyhomie.app.core.data.entity.MessageEntity
import com.heyhomie.app.core.model.ChatMessage
import com.heyhomie.app.core.model.MessageRole
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.map
import javax.inject.Inject
import javax.inject.Singleton

@Singleton
class ChatRepository @Inject constructor(
    private val apiClient: HomieApiClient,
    private val messageDao: MessageDao
) {
    fun getMessages(conversationId: String): Flow<List<ChatMessage>> =
        messageDao.getMessages(conversationId).map { entities -> entities.map { it.toChatMessage() } }
    fun getConversationIds(): Flow<List<String>> = messageDao.getConversationIds()
    suspend fun sendMessage(text: String, conversationId: String): ChatMessage {
        messageDao.insert(MessageEntity(conversationId = conversationId, role = "user", text = text))
        val responseText = apiClient.sendMessage(text, conversationId)
        val entity = MessageEntity(conversationId = conversationId, role = "assistant", text = responseText)
        messageDao.insert(entity)
        return entity.toChatMessage()
    }
    fun streamMessages(conversationId: String) = apiClient.streamMessages(conversationId)
    suspend fun getBriefing(): String = apiClient.getBriefing()
    suspend fun saveMessage(conversationId: String, role: String, text: String) {
        messageDao.insert(MessageEntity(conversationId = conversationId, role = role, text = text))
    }
}
private fun MessageEntity.toChatMessage() = ChatMessage(
    id = id,
    role = when (role) { "user" -> MessageRole.USER; "assistant" -> MessageRole.ASSISTANT; else -> MessageRole.SYSTEM },
    text = text, timestamp = timestamp
)
