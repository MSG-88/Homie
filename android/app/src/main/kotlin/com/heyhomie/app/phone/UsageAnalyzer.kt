package com.heyhomie.app.phone

import android.app.usage.UsageStatsManager
import android.content.Context
import android.content.pm.PackageManager
import dagger.hilt.android.qualifiers.ApplicationContext
import javax.inject.Inject
import javax.inject.Singleton

data class AppUsageEntry(
    val packageName: String,
    val appName: String,
    val totalTimeMs: Long,
    val lastUsed: Long,
    val launchCount: Int
) {
    val totalTimeMinutes: Int get() = (totalTimeMs / 60_000).toInt()
}

@Singleton
class UsageAnalyzer @Inject constructor(
    @ApplicationContext private val context: Context
) {
    fun getDailyUsage(): List<AppUsageEntry> {
        val usm = context.getSystemService(Context.USAGE_STATS_SERVICE) as? UsageStatsManager
            ?: return emptyList()
        val endTime = System.currentTimeMillis()
        val startTime = endTime - 24 * 60 * 60 * 1000
        val stats = usm.queryUsageStats(UsageStatsManager.INTERVAL_DAILY, startTime, endTime)
        val pm = context.packageManager
        return stats
            .filter { it.totalTimeInForeground > 0 }
            .map { stat ->
                val appName = try {
                    pm.getApplicationLabel(
                        pm.getApplicationInfo(stat.packageName, 0)
                    ).toString()
                } catch (_: PackageManager.NameNotFoundException) { stat.packageName }
                AppUsageEntry(
                    packageName = stat.packageName,
                    appName = appName,
                    totalTimeMs = stat.totalTimeInForeground,
                    lastUsed = stat.lastTimeUsed,
                    launchCount = 0
                )
            }
            .sortedByDescending { it.totalTimeMs }
    }

    fun getTotalScreenTimeMinutes(): Int = getDailyUsage().sumOf { it.totalTimeMinutes }
}
