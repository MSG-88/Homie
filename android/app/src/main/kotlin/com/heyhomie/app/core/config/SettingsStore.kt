package com.heyhomie.app.core.config

import android.content.Context
import androidx.datastore.preferences.core.booleanPreferencesKey
import androidx.datastore.preferences.core.edit
import androidx.datastore.preferences.core.intPreferencesKey
import androidx.datastore.preferences.core.stringPreferencesKey
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
        val SERVER_URL = stringPreferencesKey("server_url")
        val DARK_THEME = booleanPreferencesKey("dark_theme")
        val NOTIFICATIONS_ENABLED = booleanPreferencesKey("notifications_enabled")
        val BRIEFING_ENABLED = booleanPreferencesKey("briefing_enabled")
        val LAST_CONNECTED_HOST = stringPreferencesKey("last_connected_host")
        val LAST_CONNECTED_PORT = intPreferencesKey("last_connected_port")
    }

    val scanlines: Flow<Boolean> = context.dataStore.data.map { it[SCANLINES_ENABLED] ?: true }
    val highContrast: Flow<Boolean> = context.dataStore.data.map { it[HIGH_CONTRAST] ?: false }
    val soundEffects: Flow<Boolean> = context.dataStore.data.map { it[SOUND_EFFECTS] ?: true }
    val qubridApiKey: Flow<String> = context.dataStore.data.map { it[QUBRID_API_KEY] ?: "" }
    val syncScope: Flow<String> = context.dataStore.data.map { it[SYNC_SCOPE] ?: "all" }
    val serverUrl: Flow<String> = context.dataStore.data.map { it[SERVER_URL] ?: "" }
    val darkTheme: Flow<Boolean> = context.dataStore.data.map { it[DARK_THEME] ?: true }
    val notificationsEnabled: Flow<Boolean> = context.dataStore.data.map { it[NOTIFICATIONS_ENABLED] ?: true }
    val briefingEnabled: Flow<Boolean> = context.dataStore.data.map { it[BRIEFING_ENABLED] ?: true }
    val lastConnectedHost: Flow<String> = context.dataStore.data.map { it[LAST_CONNECTED_HOST] ?: "" }
    val lastConnectedPort: Flow<Int> = context.dataStore.data.map { it[LAST_CONNECTED_PORT] ?: 3141 }

    suspend fun setScanlines(enabled: Boolean) { context.dataStore.edit { it[SCANLINES_ENABLED] = enabled } }
    suspend fun setHighContrast(enabled: Boolean) { context.dataStore.edit { it[HIGH_CONTRAST] = enabled } }
    suspend fun setSoundEffects(enabled: Boolean) { context.dataStore.edit { it[SOUND_EFFECTS] = enabled } }
    suspend fun setQubridApiKey(key: String) { context.dataStore.edit { it[QUBRID_API_KEY] = key } }
    suspend fun setSyncScope(scope: String) { context.dataStore.edit { it[SYNC_SCOPE] = scope } }
    suspend fun setServerUrl(url: String) { context.dataStore.edit { it[SERVER_URL] = url } }
    suspend fun setDarkTheme(enabled: Boolean) { context.dataStore.edit { it[DARK_THEME] = enabled } }
    suspend fun setNotificationsEnabled(enabled: Boolean) { context.dataStore.edit { it[NOTIFICATIONS_ENABLED] = enabled } }
    suspend fun setBriefingEnabled(enabled: Boolean) { context.dataStore.edit { it[BRIEFING_ENABLED] = enabled } }
    suspend fun setLastConnected(host: String, port: Int) {
        context.dataStore.edit { it[LAST_CONNECTED_HOST] = host; it[LAST_CONNECTED_PORT] = port }
    }
}
