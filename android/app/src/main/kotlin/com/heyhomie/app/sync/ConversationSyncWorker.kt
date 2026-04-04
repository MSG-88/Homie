package com.heyhomie.app.sync

import android.content.Context
import androidx.hilt.work.HiltWorker
import androidx.work.CoroutineWorker
import androidx.work.WorkerParameters
import com.heyhomie.app.core.api.HomieApiClient
import com.heyhomie.app.core.config.SettingsStore
import com.heyhomie.app.core.data.dao.MessageDao
import dagger.assisted.Assisted
import dagger.assisted.AssistedInject
import kotlinx.coroutines.flow.first

@HiltWorker
class ConversationSyncWorker @AssistedInject constructor(
    @Assisted appContext: Context,
    @Assisted workerParams: WorkerParameters,
    private val apiClient: HomieApiClient,
    private val messageDao: MessageDao,
    private val settingsStore: SettingsStore
) : CoroutineWorker(appContext, workerParams) {
    override suspend fun doWork(): Result = try {
        val serverUrl = settingsStore.serverUrl.first()
        if (serverUrl.isBlank()) Result.success()
        else {
            apiClient.configure(serverUrl)
            if (!apiClient.healthCheck()) Result.retry() else Result.success()
        }
    } catch (_: Exception) { Result.retry() }
}
