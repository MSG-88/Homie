package com.heyhomie.app.core.config

import android.content.Context
import androidx.datastore.preferences.core.*
import androidx.datastore.preferences.preferencesDataStore
import dagger.hilt.android.qualifiers.ApplicationContext
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.map
import javax.inject.Inject
import javax.inject.Singleton

private val Context.dataStore by preferencesDataStore(name = "homie_settings")

@Singleton
class SettingsStore @Inject constructor(
    @ApplicationContext private val context: Context
) {
    companion object {
        val SCANLINES_ENABLED = booleanPreferencesKey("scanlines_enabled")
        val HIGH_CONTRAST = booleanPreferencesKey("high_contrast")
        val SOUND_EFFECTS = booleanPreferencesKey("sound_effects")
        val QUBRID_API_KEY = stringPreferencesKey("qubrid_api_key")
        val INFERENCE_PRIORITY = stringPreferencesKey("inference_priority")
        val SYNC_SCOPE = stringPreferencesKey("sync_scope")
        val AUTO_DISCOVER = booleanPreferencesKey("auto_discover")
    }

    val scanlines: Flow<Boolean> = context.dataStore.data.map { it[SCANLINES_ENABLED] ?: true }
    val highContrast: Flow<Boolean> = context.dataStore.data.map { it[HIGH_CONTRAST] ?: false }
    val soundEffects: Flow<Boolean> = context.dataStore.data.map { it[SOUND_EFFECTS] ?: true }
    val qubridApiKey: Flow<String> = context.dataStore.data.map { it[QUBRID_API_KEY] ?: "" }
    val syncScope: Flow<String> = context.dataStore.data.map { it[SYNC_SCOPE] ?: "all" }

    suspend fun setScanlines(enabled: Boolean) {
        context.dataStore.edit { it[SCANLINES_ENABLED] = enabled }
    }
    suspend fun setHighContrast(enabled: Boolean) {
        context.dataStore.edit { it[HIGH_CONTRAST] = enabled }
    }
    suspend fun setSoundEffects(enabled: Boolean) {
        context.dataStore.edit { it[SOUND_EFFECTS] = enabled }
    }
    suspend fun setQubridApiKey(key: String) {
        context.dataStore.edit { it[QUBRID_API_KEY] = key }
    }
    suspend fun setSyncScope(scope: String) {
        context.dataStore.edit { it[SYNC_SCOPE] = scope }
    }
}
