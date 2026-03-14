package com.heyhomie.app.phone

import android.app.ActivityManager
import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.os.BatteryManager
import android.os.Build
import android.os.Environment
import android.os.StatFs
import android.view.WindowManager
import dagger.hilt.android.qualifiers.ApplicationContext
import javax.inject.Inject
import javax.inject.Singleton

@Singleton
class DeviceProfiler @Inject constructor(
    @ApplicationContext private val context: Context
) {
    fun profile(): DeviceProfile {
        val activityManager = context.getSystemService(Context.ACTIVITY_SERVICE) as ActivityManager
        val memInfo = ActivityManager.MemoryInfo()
        activityManager.getMemoryInfo(memInfo)

        val stat = StatFs(Environment.getDataDirectory().path)

        val batteryIntent = context.registerReceiver(null, IntentFilter(Intent.ACTION_BATTERY_CHANGED))
        val batteryLevel = batteryIntent?.getIntExtra(BatteryManager.EXTRA_LEVEL, -1) ?: -1
        val batteryScale = batteryIntent?.getIntExtra(BatteryManager.EXTRA_SCALE, 100) ?: 100
        val isCharging = batteryIntent?.getIntExtra(BatteryManager.EXTRA_PLUGGED, 0) != 0

        val sensorManager = context.getSystemService(Context.SENSOR_SERVICE) as android.hardware.SensorManager
        val sensors = sensorManager.getSensorList(android.hardware.Sensor.TYPE_ALL).map { it.name }

        val wm = context.getSystemService(Context.WINDOW_SERVICE) as WindowManager
        val display = wm.defaultDisplay
        val refreshRate = display.refreshRate

        return DeviceProfile(
            cpuCores = Runtime.getRuntime().availableProcessors(),
            cpuArch = Build.SUPPORTED_ABIS.firstOrNull() ?: "unknown",
            totalRamMb = memInfo.totalMem / (1024 * 1024),
            availableRamMb = memInfo.availMem / (1024 * 1024),
            totalStorageMb = stat.totalBytes / (1024 * 1024),
            freeStorageMb = stat.availableBytes / (1024 * 1024),
            batteryLevel = if (batteryScale > 0) (batteryLevel * 100) / batteryScale else -1,
            isCharging = isCharging,
            screenDensity = context.resources.displayMetrics.densityDpi,
            refreshRate = refreshRate,
            gpuRenderer = "unknown",
            supportsVulkan = Build.VERSION.SDK_INT >= 24,
            sensors = sensors
        )
    }
}
