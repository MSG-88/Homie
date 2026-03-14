package com.heyhomie.app.core.inference.di

import com.heyhomie.app.core.config.SettingsStore
import com.heyhomie.app.core.inference.InferenceRouter
import com.heyhomie.app.core.inference.QubridClient
import dagger.Module
import dagger.Provides
import dagger.hilt.InstallIn
import dagger.hilt.components.SingletonComponent
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.runBlocking
import javax.inject.Singleton

@Module
@InstallIn(SingletonComponent::class)
object InferenceModule {
    @Provides @Singleton
    fun provideQubridClient(settings: SettingsStore): QubridClient {
        val apiKey = runBlocking { settings.qubridApiKey.first() }
        return QubridClient(apiKey = apiKey)
    }

    @Provides @Singleton
    fun provideInferenceRouter(qubridClient: QubridClient): InferenceRouter {
        return InferenceRouter(localBridge = null, qubridClient = qubridClient)
    }
}
