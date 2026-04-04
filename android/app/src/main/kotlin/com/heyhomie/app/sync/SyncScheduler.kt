package com.heyhomie.app.sync

import android.content.Context
import androidx.work.BackoffPolicy
import androidx.work.Constraints
import androidx.work.ExistingPeriodicWorkPolicy
import androidx.work.NetworkType
import androidx.work.PeriodicWorkRequestBuilder
import androidx.work.WorkManager
import dagger.hilt.android.qualifiers.ApplicationContext
import java.util.concurrent.TimeUnit
import javax.inject.Inject
import javax.inject.Singleton

@Singleton
class SyncScheduler @Inject constructor(@ApplicationContext private val context: Context) {
    fun scheduleConversationSync() {
        val c = Constraints.Builder().setRequiredNetworkType(NetworkType.CONNECTED).build()
        val req = PeriodicWorkRequestBuilder<ConversationSyncWorker>(15, TimeUnit.MINUTES)
            .setConstraints(c).setBackoffCriteria(BackoffPolicy.EXPONENTIAL, 5, TimeUnit.MINUTES).build()
        WorkManager.getInstance(context).enqueueUniquePeriodicWork("conversation_sync", ExistingPeriodicWorkPolicy.KEEP, req)
    }
    fun scheduleBriefingSync() {
        val c = Constraints.Builder().setRequiredNetworkType(NetworkType.CONNECTED).build()
        val req = PeriodicWorkRequestBuilder<BriefingSyncWorker>(6, TimeUnit.HOURS).setConstraints(c).build()
        WorkManager.getInstance(context).enqueueUniquePeriodicWork("briefing_sync", ExistingPeriodicWorkPolicy.KEEP, req)
    }
}
