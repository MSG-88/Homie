package com.heyhomie.app.notifications

import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import javax.inject.Inject
import javax.inject.Singleton

data class CapturedNotification(
    val packageName: String,
    val appName: String,
    val title: String,
    val text: String,
    val timestamp: Long,
    val category: String? = null,
    val priority: Int = 0
)

@Singleton
class NotificationStore @Inject constructor() {
    private val _notifications = MutableStateFlow<List<CapturedNotification>>(emptyList())
    val notifications: StateFlow<List<CapturedNotification>> = _notifications

    fun add(notification: CapturedNotification) {
        _notifications.value = (_notifications.value + notification).takeLast(200)
    }

    fun getUnread(since: Long): List<CapturedNotification> =
        _notifications.value.filter { it.timestamp > since }

    fun summarize(): String {
        val recent = getUnread(System.currentTimeMillis() - 3600_000)
        if (recent.isEmpty()) return "No new notifications in the last hour."
        val byApp = recent.groupBy { it.appName }
        val parts = byApp.map { (app, notifs) -> "$app: ${notifs.size}" }
        return "You got ${recent.size} notifications in the last hour - ${parts.joinToString(", ")}"
    }

    fun clear() { _notifications.value = emptyList() }
}
