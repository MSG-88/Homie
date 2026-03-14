package com.heyhomie.app.notifications

import android.service.notification.NotificationListenerService
import android.service.notification.StatusBarNotification

class HomieNotificationListener : NotificationListenerService() {

    companion object {
        var store: NotificationStore? = null
    }

    override fun onNotificationPosted(sbn: StatusBarNotification) {
        val extras = sbn.notification.extras
        val captured = CapturedNotification(
            packageName = sbn.packageName,
            appName = extras.getCharSequence("android.title.big")?.toString()
                ?: sbn.packageName.substringAfterLast('.'),
            title = extras.getCharSequence("android.title")?.toString() ?: "",
            text = extras.getCharSequence("android.text")?.toString() ?: "",
            timestamp = sbn.postTime,
            category = sbn.notification.category,
            priority = sbn.notification.priority
        )
        store?.add(captured)
    }

    override fun onNotificationRemoved(sbn: StatusBarNotification) {
        // Could track dismissed notifications if needed
    }
}
