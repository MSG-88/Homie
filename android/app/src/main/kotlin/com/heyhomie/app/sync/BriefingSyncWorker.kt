package com.heyhomie.app.sync

import android.Manifest
import android.app.NotificationChannel
import android.app.NotificationManager
import android.content.Context
import android.content.pm.PackageManager
import android.os.Build
import androidx.core.app.NotificationCompat
import androidx.core.app.NotificationManagerCompat
import androidx.core.content.ContextCompat
import androidx.hilt.work.HiltWorker
import androidx.work.CoroutineWorker
import androidx.work.WorkerParameters
import com.heyhomie.app.R
import com.heyhomie.app.core.api.HomieApiClient
import com.heyhomie.app.core.config.SettingsStore
import dagger.assisted.Assisted
import dagger.assisted.AssistedInject
import kotlinx.coroutines.flow.first

@HiltWorker
class BriefingSyncWorker @AssistedInject constructor(
    @Assisted private val appContext: Context,
    @Assisted workerParams: WorkerParameters,
    private val apiClient: HomieApiClient,
    private val settingsStore: SettingsStore
) : CoroutineWorker(appContext, workerParams) {
    override suspend fun doWork(): Result = try {
        if (!settingsStore.briefingEnabled.first()) return Result.success()
        val serverUrl = settingsStore.serverUrl.first()
        if (serverUrl.isBlank()) return Result.success()
        apiClient.configure(serverUrl)
        if (!apiClient.healthCheck()) return Result.retry()
        val briefing = apiClient.getBriefing()
        if (briefing.isNotBlank()) showNotification(briefing)
        Result.success()
    } catch (_: Exception) { Result.retry() }

    private fun showNotification(briefing: String) {
        val channelId = "homie_briefing"
        (appContext.getSystemService(Context.NOTIFICATION_SERVICE) as NotificationManager)
            .createNotificationChannel(NotificationChannel(channelId, "Homie Briefings", NotificationManager.IMPORTANCE_DEFAULT))
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU &&
            ContextCompat.checkSelfPermission(appContext, Manifest.permission.POST_NOTIFICATIONS) != PackageManager.PERMISSION_GRANTED) return
        NotificationManagerCompat.from(appContext).notify(1001,
            NotificationCompat.Builder(appContext, channelId).setSmallIcon(R.mipmap.ic_launcher)
                .setContentTitle("Homie Briefing").setContentText(briefing)
                .setStyle(NotificationCompat.BigTextStyle().bigText(briefing))
                .setPriority(NotificationCompat.PRIORITY_DEFAULT).setAutoCancel(true).build())
    }
}
