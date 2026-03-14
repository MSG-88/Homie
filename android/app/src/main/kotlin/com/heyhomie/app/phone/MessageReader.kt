package com.heyhomie.app.phone

import android.content.Context
import android.net.Uri
import dagger.hilt.android.qualifiers.ApplicationContext
import javax.inject.Inject
import javax.inject.Singleton

data class SmsMessage(
    val address: String,
    val body: String,
    val date: Long,
    val type: Int
) {
    val isIncoming: Boolean get() = type == 1
}

@Singleton
class MessageReader @Inject constructor(
    @ApplicationContext private val context: Context
) {
    fun getRecentMessages(limit: Int = 50): List<SmsMessage> {
        val messages = mutableListOf<SmsMessage>()
        val cursor = context.contentResolver.query(
            Uri.parse("content://sms/inbox"),
            arrayOf("address", "body", "date", "type"),
            null, null, "date DESC"
        ) ?: return emptyList()

        cursor.use {
            var count = 0
            while (it.moveToNext() && count < limit) {
                messages.add(
                    SmsMessage(
                        address = it.getString(0) ?: "",
                        body = it.getString(1) ?: "",
                        date = it.getLong(2),
                        type = it.getInt(3)
                    )
                )
                count++
            }
        }
        return messages
    }

    fun searchMessages(query: String): List<SmsMessage> {
        val cursor = context.contentResolver.query(
            Uri.parse("content://sms"),
            arrayOf("address", "body", "date", "type"),
            "body LIKE ?", arrayOf("%$query%"), "date DESC"
        ) ?: return emptyList()

        val results = mutableListOf<SmsMessage>()
        cursor.use {
            while (it.moveToNext()) {
                results.add(
                    SmsMessage(
                        address = it.getString(0) ?: "",
                        body = it.getString(1) ?: "",
                        date = it.getLong(2),
                        type = it.getInt(3)
                    )
                )
            }
        }
        return results
    }
}
