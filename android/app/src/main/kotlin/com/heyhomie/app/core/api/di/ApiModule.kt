package com.heyhomie.app.core.api.di

import com.heyhomie.app.core.api.HomieApiClient
import com.heyhomie.app.core.config.SettingsStore
import dagger.Module
import dagger.Provides
import dagger.hilt.InstallIn
import dagger.hilt.components.SingletonComponent
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.runBlocking
import javax.inject.Singleton

@Module
@InstallIn(SingletonComponent::class)
object ApiModule {
    @Provides @Singleton
    fun provideHomieApiClient(settings: SettingsStore): HomieApiClient {
        val client = HomieApiClient()
        val serverUrl = runBlocking { settings.serverUrl.first() }
        if (serverUrl.isNotBlank()) client.configure(serverUrl)
        return client
    }
}
