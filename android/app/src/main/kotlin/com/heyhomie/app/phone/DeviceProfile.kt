package com.heyhomie.app.phone

data class DeviceProfile(
    val cpuCores: Int,
    val cpuArch: String,
    val totalRamMb: Long,
    val availableRamMb: Long,
    val totalStorageMb: Long,
    val freeStorageMb: Long,
    val batteryLevel: Int,
    val isCharging: Boolean,
    val screenDensity: Int,
    val refreshRate: Float,
    val gpuRenderer: String,
    val supportsVulkan: Boolean,
    val sensors: List<String> = emptyList(),
    val networkType: String = "unknown",
    val signalStrength: Int = 0
) {
    val capabilityScore: Int
        get() {
            var score = 0
            score += when {
                totalRamMb >= 8192 -> 35
                totalRamMb >= 6144 -> 25
                totalRamMb >= 4096 -> 15
                else -> 5
            }
            score += when {
                cpuCores >= 8 -> 20
                cpuCores >= 6 -> 15
                cpuCores >= 4 -> 10
                else -> 5
            }
            score += if (cpuArch.contains("arm64") || cpuArch.contains("v8a")) 15 else 5
            score += if (supportsVulkan) 15 else 5
            score += if (freeStorageMb >= 4000) 15 else 5
            return score.coerceIn(0, 100)
        }

    val recommendedModelSize: String
        get() = when {
            capabilityScore >= 70 && totalRamMb >= 8192 -> "7B Q4"
            capabilityScore >= 50 && totalRamMb >= 6144 -> "3B Q4"
            else -> "1.5B Q4"
        }
}
