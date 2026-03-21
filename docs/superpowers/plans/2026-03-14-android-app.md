# Android App Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a native Android app for Homie AI with retro pixel theme, on-device inference, phone analysis, and LAN sync with desktop.

**Architecture:** Kotlin/Jetpack Compose single-activity app with MVVM. Room DB for persistence, llama.cpp via NDK/JNI for local inference, Qubrid cloud fallback. WebSocket-based LAN sync with paired desktop. Retro 8-bit pixel theme throughout.

**Tech Stack:** Kotlin, Jetpack Compose, Material3, Room, DataStore, llama.cpp (NDK/CMake), OkHttp, Hilt, WorkManager, NSD (Network Service Discovery)

---

## Chunk 1: Project Scaffolding & Theme

### Task 1: Gradle Project Setup

**Files:**
- Create: `android/settings.gradle.kts`
- Create: `android/build.gradle.kts`
- Create: `android/app/build.gradle.kts`
- Create: `android/gradle.properties`
- Create: `android/app/src/main/AndroidManifest.xml`
- Create: `android/app/src/main/kotlin/com/heyhomie/app/HomieApp.kt`
- Create: `android/app/src/main/kotlin/com/heyhomie/app/MainActivity.kt`

- [ ] **Step 1: Create root Gradle files**

`android/settings.gradle.kts`:
```kotlin
pluginManagement {
    repositories {
        google()
        mavenCentral()
        gradlePluginPortal()
    }
}
dependencyResolutionManagement {
    repositoriesMode.set(RepositoriesMode.FAIL_ON_PROJECT_REPOS)
    repositories {
        google()
        mavenCentral()
    }
}
rootProject.name = "HomieAI"
include(":app")
```

`android/build.gradle.kts`:
```kotlin
plugins {
    id("com.android.application") version "8.7.0" apply false
    id("org.jetbrains.kotlin.android") version "2.1.0" apply false
    id("com.google.devtools.ksp") version "2.1.0-1.0.29" apply false
    id("com.google.dagger.hilt.android") version "2.51.1" apply false
    id("org.jetbrains.kotlin.plugin.compose") version "2.1.0" apply false
}
```

`android/gradle.properties`:
```properties
android.useAndroidX=true
kotlin.code.style=official
android.nonTransitiveRClass=true
```

- [ ] **Step 2: Create app build.gradle.kts**

`android/app/build.gradle.kts`:
```kotlin
plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
    id("com.google.devtools.ksp")
    id("com.google.dagger.hilt.android")
    id("org.jetbrains.kotlin.plugin.compose")
}

android {
    namespace = "com.heyhomie.app"
    compileSdk = 35

    defaultConfig {
        applicationId = "com.heyhomie.app"
        minSdk = 28
        targetSdk = 35
        versionCode = 1
        versionName = "0.1.0"
        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"
    }
    buildFeatures { compose = true }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
    kotlinOptions { jvmTarget = "17" }
}

dependencies {
    // Compose
    val composeBom = platform("androidx.compose:compose-bom:2024.12.01")
    implementation(composeBom)
    implementation("androidx.compose.ui:ui")
    implementation("androidx.compose.material3:material3")
    implementation("androidx.compose.ui:ui-tooling-preview")
    implementation("androidx.activity:activity-compose:1.9.3")
    implementation("androidx.navigation:navigation-compose:2.8.5")
    debugImplementation("androidx.compose.ui:ui-tooling")

    // Lifecycle
    implementation("androidx.lifecycle:lifecycle-viewmodel-compose:2.8.7")
    implementation("androidx.lifecycle:lifecycle-runtime-compose:2.8.7")

    // Room
    implementation("androidx.room:room-runtime:2.6.1")
    implementation("androidx.room:room-ktx:2.6.1")
    ksp("androidx.room:room-compiler:2.6.1")

    // DataStore
    implementation("androidx.datastore:datastore-preferences:1.1.1")

    // Hilt
    implementation("com.google.dagger:hilt-android:2.51.1")
    ksp("com.google.dagger:hilt-compiler:2.51.1")
    implementation("androidx.hilt:hilt-navigation-compose:1.2.0")

    // WorkManager
    implementation("androidx.work:work-runtime-ktx:2.10.0")

    // Network
    implementation("com.squareup.okhttp3:okhttp:4.12.0")

    // Testing
    testImplementation("junit:junit:4.13.2")
    testImplementation("org.jetbrains.kotlinx:kotlinx-coroutines-test:1.9.0")
    testImplementation("io.mockk:mockk:1.13.13")
    androidTestImplementation(composeBom)
    androidTestImplementation("androidx.compose.ui:ui-test-junit4")
    androidTestImplementation("androidx.test.ext:junit:1.2.1")
}
```

- [ ] **Step 3: Create AndroidManifest.xml**

```xml
<?xml version="1.0" encoding="utf-8"?>
<manifest xmlns:android="http://schemas.android.com/apk/res/android">

    <uses-permission android:name="android.permission.INTERNET" />
    <uses-permission android:name="android.permission.ACCESS_NETWORK_STATE" />
    <uses-permission android:name="android.permission.ACCESS_WIFI_STATE" />
    <uses-permission android:name="android.permission.POST_NOTIFICATIONS" />
    <uses-permission android:name="android.permission.READ_SMS" />
    <uses-permission android:name="android.permission.READ_CONTACTS" />
    <uses-permission android:name="android.permission.RECORD_AUDIO" />
    <uses-permission android:name="android.permission.PACKAGE_USAGE_STATS"
        tools:ignore="ProtectedPermissions" />
    <uses-permission android:name="android.permission.FOREGROUND_SERVICE" />
    <uses-permission android:name="android.permission.RECEIVE_BOOT_COMPLETED" />

    <application
        android:name=".HomieApp"
        android:label="Homie AI"
        android:icon="@mipmap/ic_launcher"
        android:theme="@style/Theme.HomieAI"
        android:supportsRtl="true">

        <activity
            android:name=".MainActivity"
            android:exported="true"
            android:windowSoftInputMode="adjustResize">
            <intent-filter>
                <action android:name="android.intent.action.MAIN" />
                <category android:name="android.intent.category.LAUNCHER" />
            </intent-filter>
        </activity>

    </application>
</manifest>
```

- [ ] **Step 4: Create Application and MainActivity**

`HomieApp.kt`:
```kotlin
package com.heyhomie.app

import android.app.Application
import dagger.hilt.android.HiltAndroidApp

@HiltAndroidApp
class HomieApp : Application()
```

`MainActivity.kt`:
```kotlin
package com.heyhomie.app

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import com.heyhomie.app.ui.theme.HomieRetroTheme
import com.heyhomie.app.ui.HomieNavHost
import dagger.hilt.android.AndroidEntryPoint

@AndroidEntryPoint
class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        setContent {
            HomieRetroTheme {
                HomieNavHost()
            }
        }
    }
}
```

- [ ] **Step 5: Verify project builds**

Run: `cd android && ./gradlew assembleDebug`
Expected: BUILD SUCCESSFUL

- [ ] **Step 6: Commit**

```bash
git add android/
git commit -m "feat(android): scaffold Kotlin/Compose project with Gradle setup"
```

---

### Task 2: Retro Pixel Theme — Colors & Typography

**Files:**
- Create: `android/app/src/main/kotlin/com/heyhomie/app/ui/theme/Color.kt`
- Create: `android/app/src/main/kotlin/com/heyhomie/app/ui/theme/Type.kt`
- Create: `android/app/src/main/kotlin/com/heyhomie/app/ui/theme/Theme.kt`
- Test: `android/app/src/test/kotlin/com/heyhomie/app/ui/theme/ThemeTest.kt`

- [ ] **Step 1: Write theme color test**

```kotlin
package com.heyhomie.app.ui.theme

import org.junit.Assert.*
import org.junit.Test

class ThemeTest {
    @Test
    fun `retro colors have correct hex values`() {
        assertEquals(0xFF0D0D0D.toInt(), RetroDark.toArgb())
        assertEquals(0xFF39FF14.toInt(), RetroGreen.toArgb())
        assertEquals(0xFFFFB000.toInt(), RetroAmber.toArgb())
        assertEquals(0xFF00E5FF.toInt(), RetroCyan.toArgb())
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd android && ./gradlew testDebugUnitTest --tests "*.ThemeTest"`
Expected: FAIL — colors not defined

- [ ] **Step 3: Create Color.kt**

```kotlin
package com.heyhomie.app.ui.theme

import androidx.compose.ui.graphics.Color

// Retro 8-bit palette
val RetroDark = Color(0xFF0D0D0D)
val RetroDarkSurface = Color(0xFF1A1A2E)
val RetroDarkCard = Color(0xFF16213E)
val RetroGreen = Color(0xFF39FF14)
val RetroAmber = Color(0xFFFFB000)
val RetroCyan = Color(0xFF00E5FF)
val RetroRed = Color(0xFFFF073A)
val RetroWhite = Color(0xFFE0E0E0)
val RetroGray = Color(0xFF888888)
val RetroDimGreen = Color(0xFF1B5E20)
```

- [ ] **Step 4: Create Type.kt with pixel font**

```kotlin
package com.heyhomie.app.ui.theme

import androidx.compose.material3.Typography
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.text.font.Font
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.sp
import com.heyhomie.app.R

val PressStart2P = FontFamily(
    Font(R.font.press_start_2p, FontWeight.Normal)
)

val JetBrainsMono = FontFamily(
    Font(R.font.jetbrains_mono_regular, FontWeight.Normal),
    Font(R.font.jetbrains_mono_bold, FontWeight.Bold)
)

val RetroTypography = Typography(
    displayLarge = TextStyle(
        fontFamily = PressStart2P, fontSize = 24.sp, color = RetroGreen
    ),
    headlineMedium = TextStyle(
        fontFamily = PressStart2P, fontSize = 16.sp, color = RetroGreen
    ),
    titleMedium = TextStyle(
        fontFamily = PressStart2P, fontSize = 12.sp, color = RetroAmber
    ),
    bodyLarge = TextStyle(
        fontFamily = JetBrainsMono, fontSize = 16.sp, color = RetroWhite
    ),
    bodyMedium = TextStyle(
        fontFamily = JetBrainsMono, fontSize = 14.sp, color = RetroWhite
    ),
    labelMedium = TextStyle(
        fontFamily = PressStart2P, fontSize = 10.sp, color = RetroCyan
    )
)
```

- [ ] **Step 5: Create Theme.kt**

```kotlin
package com.heyhomie.app.ui.theme

import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.darkColorScheme
import androidx.compose.runtime.Composable

private val RetroColorScheme = darkColorScheme(
    primary = RetroGreen,
    secondary = RetroAmber,
    tertiary = RetroCyan,
    background = RetroDark,
    surface = RetroDarkSurface,
    surfaceVariant = RetroDarkCard,
    onPrimary = RetroDark,
    onSecondary = RetroDark,
    onTertiary = RetroDark,
    onBackground = RetroWhite,
    onSurface = RetroWhite,
    error = RetroRed,
    onError = RetroDark
)

@Composable
fun HomieRetroTheme(content: @Composable () -> Unit) {
    MaterialTheme(
        colorScheme = RetroColorScheme,
        typography = RetroTypography,
        content = content
    )
}
```

- [ ] **Step 6: Add font files placeholder**

Download Press Start 2P and JetBrains Mono to `android/app/src/main/res/font/`:
- `press_start_2p.ttf`
- `jetbrains_mono_regular.ttf`
- `jetbrains_mono_bold.ttf`

- [ ] **Step 7: Run test to verify it passes**

Run: `cd android && ./gradlew testDebugUnitTest --tests "*.ThemeTest"`
Expected: PASS

- [ ] **Step 8: Commit**

```bash
git add android/app/src/main/kotlin/com/heyhomie/app/ui/theme/
git add android/app/src/main/res/font/
git add android/app/src/test/
git commit -m "feat(android): add retro pixel theme with 8-bit colors and pixel fonts"
```

---

### Task 3: PixelBorder Modifier & Retro UI Components

**Files:**
- Create: `android/app/src/main/kotlin/com/heyhomie/app/ui/components/PixelBorder.kt`
- Create: `android/app/src/main/kotlin/com/heyhomie/app/ui/components/RetroCard.kt`
- Create: `android/app/src/main/kotlin/com/heyhomie/app/ui/components/ScanlineOverlay.kt`
- Create: `android/app/src/main/kotlin/com/heyhomie/app/ui/components/CrtGlow.kt`
- Create: `android/app/src/main/kotlin/com/heyhomie/app/ui/components/RetroProgressBar.kt`
- Test: `android/app/src/test/kotlin/com/heyhomie/app/ui/components/PixelBorderTest.kt`

- [ ] **Step 1: Write PixelBorder test**

```kotlin
package com.heyhomie.app.ui.components

import org.junit.Assert.*
import org.junit.Test

class PixelBorderTest {
    @Test
    fun `pixel step size defaults to 4dp equivalent`() {
        val config = PixelBorderConfig()
        assertEquals(4f, config.stepSize)
    }

    @Test
    fun `pixel border config accepts custom color`() {
        val config = PixelBorderConfig(borderColor = 0xFF39FF14)
        assertEquals(0xFF39FF14, config.borderColor)
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd android && ./gradlew testDebugUnitTest --tests "*.PixelBorderTest"`
Expected: FAIL

- [ ] **Step 3: Create PixelBorder.kt**

```kotlin
package com.heyhomie.app.ui.components

import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.drawBehind
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.unit.dp
import com.heyhomie.app.ui.theme.RetroGreen

data class PixelBorderConfig(
    val stepSize: Float = 4f,
    val borderColor: Long = 0xFF39FF14,
    val borderWidth: Float = 2f
)

fun Modifier.pixelBorder(
    color: Color = RetroGreen,
    stepSize: Float = 4f,
    width: Float = 2f
): Modifier = this.drawBehind {
    val w = size.width
    val h = size.height
    val step = stepSize * density

    // Top edge (stepped)
    var x = 0f
    while (x < w) {
        drawRect(color, Offset(x, 0f), Size(minOf(step, w - x), width * density))
        x += step
    }
    // Bottom edge
    x = 0f
    while (x < w) {
        drawRect(color, Offset(x, h - width * density), Size(minOf(step, w - x), width * density))
        x += step
    }
    // Left edge
    var y = 0f
    while (y < h) {
        drawRect(color, Offset(0f, y), Size(width * density, minOf(step, h - y)))
        y += step
    }
    // Right edge
    y = 0f
    while (y < h) {
        drawRect(color, Offset(w - width * density, y), Size(width * density, minOf(step, h - y)))
        y += step
    }
    // Corner notches (staircase effect)
    val notch = step
    // Top-left
    drawRect(Color.Transparent, Offset(0f, 0f), Size(notch, notch))
    drawRect(color, Offset(notch, 0f), Size(width * density, notch))
    drawRect(color, Offset(0f, notch), Size(notch, width * density))
}
```

- [ ] **Step 4: Create ScanlineOverlay.kt**

```kotlin
package com.heyhomie.app.ui.components

import androidx.compose.foundation.Canvas
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color

@Composable
fun ScanlineOverlay(
    modifier: Modifier = Modifier,
    lineSpacing: Float = 4f,
    alpha: Float = 0.08f
) {
    Canvas(modifier = modifier.fillMaxSize()) {
        var y = 0f
        while (y < size.height) {
            drawRect(
                color = Color.Black.copy(alpha = alpha),
                topLeft = androidx.compose.ui.geometry.Offset(0f, y),
                size = androidx.compose.ui.geometry.Size(size.width, lineSpacing / 2)
            )
            y += lineSpacing * density
        }
    }
}
```

- [ ] **Step 5: Create CrtGlow.kt**

```kotlin
package com.heyhomie.app.ui.components

import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.drawBehind
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.Paint
import androidx.compose.ui.graphics.drawscope.drawIntoCanvas
import androidx.compose.ui.graphics.toArgb

fun Modifier.crtGlow(
    color: Color = Color(0xFF39FF14),
    radius: Float = 12f
): Modifier = this.drawBehind {
    drawIntoCanvas { canvas ->
        val paint = Paint().asFrameworkPaint().apply {
            isAntiAlias = true
            setShadowLayer(radius * density, 0f, 0f, color.copy(alpha = 0.6f).toArgb())
        }
        canvas.nativeCanvas.drawRoundRect(
            0f, 0f, size.width, size.height, 4f, 4f, paint
        )
    }
}
```

- [ ] **Step 6: Create RetroCard.kt**

```kotlin
package com.heyhomie.app.ui.components

import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.padding
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import com.heyhomie.app.ui.theme.RetroDarkCard
import com.heyhomie.app.ui.theme.RetroGreen

@Composable
fun RetroCard(
    modifier: Modifier = Modifier,
    content: @Composable () -> Unit
) {
    Box(
        modifier = modifier
            .pixelBorder(color = RetroGreen)
            .crtGlow()
            .background(RetroDarkCard)
            .padding(12.dp)
    ) {
        content()
    }
}
```

- [ ] **Step 7: Create RetroProgressBar.kt**

```kotlin
package com.heyhomie.app.ui.components

import androidx.compose.foundation.Canvas
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.material3.MaterialTheme
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.unit.dp
import com.heyhomie.app.ui.theme.RetroDarkCard
import com.heyhomie.app.ui.theme.RetroGreen

@Composable
fun RetroProgressBar(
    progress: Float,
    modifier: Modifier = Modifier,
    color: Color = RetroGreen,
    backgroundColor: Color = RetroDarkCard,
    label: String? = null
) {
    Canvas(
        modifier = modifier
            .fillMaxWidth()
            .height(20.dp)
            .pixelBorder(color = color, width = 1f)
    ) {
        // Background
        drawRect(backgroundColor, size = size)

        // Filled blocks (pixel-style segments)
        val blockWidth = 8f * density
        val gap = 2f * density
        val filledWidth = size.width * progress.coerceIn(0f, 1f)
        var x = gap
        while (x + blockWidth < filledWidth) {
            drawRect(
                color,
                topLeft = Offset(x, gap),
                size = Size(blockWidth, size.height - gap * 2)
            )
            x += blockWidth + gap
        }
    }
}
```

- [ ] **Step 8: Run tests**

Run: `cd android && ./gradlew testDebugUnitTest --tests "*.PixelBorderTest"`
Expected: PASS

- [ ] **Step 9: Commit**

```bash
git add android/app/src/main/kotlin/com/heyhomie/app/ui/components/
git add android/app/src/test/
git commit -m "feat(android): add retro UI components — PixelBorder, CRT glow, scanlines, progress bar"
```

---

## Chunk 2: Navigation, Boot Screen & Chat UI

### Task 4: Navigation & Game HUD Bottom Bar

**Files:**
- Create: `android/app/src/main/kotlin/com/heyhomie/app/ui/HomieNavHost.kt`
- Create: `android/app/src/main/kotlin/com/heyhomie/app/ui/components/GameHudNavBar.kt`
- Create: `android/app/src/main/kotlin/com/heyhomie/app/ui/navigation/Screen.kt`

- [ ] **Step 1: Create Screen.kt navigation routes**

```kotlin
package com.heyhomie.app.ui.navigation

sealed class Screen(val route: String, val label: String, val icon: String) {
    data object Chat : Screen("chat", "CHAT", "💬")
    data object PhoneStats : Screen("phone_stats", "STATS", "📱")
    data object Network : Screen("network", "LAN", "🌐")
    data object Settings : Screen("settings", "CONFIG", "⚙️")
}

val bottomNavScreens = listOf(Screen.Chat, Screen.PhoneStats, Screen.Network, Screen.Settings)
```

- [ ] **Step 2: Create GameHudNavBar.kt**

```kotlin
package com.heyhomie.app.ui.components

import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import com.heyhomie.app.ui.navigation.Screen
import com.heyhomie.app.ui.navigation.bottomNavScreens
import com.heyhomie.app.ui.theme.*

@Composable
fun GameHudNavBar(
    currentRoute: String?,
    onNavigate: (Screen) -> Unit,
    modifier: Modifier = Modifier
) {
    Row(
        modifier = modifier
            .fillMaxWidth()
            .background(RetroDark)
            .pixelBorder(color = RetroGreen, width = 1f)
            .padding(vertical = 4.dp),
        horizontalArrangement = Arrangement.SpaceEvenly
    ) {
        bottomNavScreens.forEach { screen ->
            val selected = currentRoute == screen.route
            val color = if (selected) RetroGreen else RetroGray
            Column(
                horizontalAlignment = Alignment.CenterHorizontally,
                modifier = Modifier
                    .clickable { onNavigate(screen) }
                    .padding(horizontal = 8.dp, vertical = 4.dp)
            ) {
                Text(
                    text = screen.icon,
                    style = RetroTypography.bodyLarge,
                    color = color
                )
                Text(
                    text = screen.label,
                    style = RetroTypography.labelMedium,
                    color = color,
                    textAlign = TextAlign.Center
                )
                if (selected) {
                    Spacer(Modifier.height(2.dp))
                    Box(
                        Modifier
                            .width(24.dp)
                            .height(2.dp)
                            .background(RetroGreen)
                    )
                }
            }
        }
    }
}
```

- [ ] **Step 3: Create HomieNavHost.kt**

```kotlin
package com.heyhomie.app.ui

import androidx.compose.foundation.layout.padding
import androidx.compose.material3.Scaffold
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.ui.Modifier
import androidx.navigation.compose.NavHost
import androidx.navigation.compose.composable
import androidx.navigation.compose.currentBackStackEntryAsState
import androidx.navigation.compose.rememberNavController
import com.heyhomie.app.ui.components.GameHudNavBar
import com.heyhomie.app.ui.navigation.Screen
import com.heyhomie.app.ui.screens.*

@Composable
fun HomieNavHost() {
    val navController = rememberNavController()
    val navBackStackEntry by navController.currentBackStackEntryAsState()
    val currentRoute = navBackStackEntry?.destination?.route

    Scaffold(
        bottomBar = {
            GameHudNavBar(
                currentRoute = currentRoute,
                onNavigate = { screen ->
                    navController.navigate(screen.route) {
                        popUpTo(Screen.Chat.route) { saveState = true }
                        launchSingleTop = true
                        restoreState = true
                    }
                }
            )
        }
    ) { innerPadding ->
        NavHost(
            navController = navController,
            startDestination = Screen.Chat.route,
            modifier = Modifier.padding(innerPadding)
        ) {
            composable(Screen.Chat.route) { ChatScreen() }
            composable(Screen.PhoneStats.route) { PhoneStatsScreen() }
            composable(Screen.Network.route) { NetworkScreen() }
            composable(Screen.Settings.route) { SettingsScreen() }
        }
    }
}
```

- [ ] **Step 4: Create placeholder screens**

Create stub composables in `android/app/src/main/kotlin/com/heyhomie/app/ui/screens/`:
- `ChatScreen.kt` — placeholder Text("CHAT")
- `PhoneStatsScreen.kt` — placeholder Text("STATS")
- `NetworkScreen.kt` — placeholder Text("NETWORK")
- `SettingsScreen.kt` — placeholder Text("SETTINGS")

Each follows this pattern:
```kotlin
package com.heyhomie.app.ui.screens

import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import com.heyhomie.app.ui.theme.RetroTypography

@Composable
fun ChatScreen() {
    Box(Modifier.fillMaxSize(), contentAlignment = Alignment.Center) {
        Text("CHAT", style = RetroTypography.headlineMedium)
    }
}
```

- [ ] **Step 5: Build and verify**

Run: `cd android && ./gradlew assembleDebug`
Expected: BUILD SUCCESSFUL

- [ ] **Step 6: Commit**

```bash
git add android/app/src/main/kotlin/com/heyhomie/app/ui/
git commit -m "feat(android): add game HUD navigation bar and screen routing"
```

---

### Task 5: Boot Screen with Retro Startup Sequence

**Files:**
- Create: `android/app/src/main/kotlin/com/heyhomie/app/ui/screens/BootScreen.kt`
- Modify: `android/app/src/main/kotlin/com/heyhomie/app/ui/HomieNavHost.kt`

- [ ] **Step 1: Create BootScreen.kt**

```kotlin
package com.heyhomie.app.ui.screens

import androidx.compose.animation.core.*
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.material3.Text
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import com.heyhomie.app.ui.components.ScanlineOverlay
import com.heyhomie.app.ui.theme.*
import kotlinx.coroutines.delay

private val bootLines = listOf(
    "HOMIE AI v0.1.0",
    "==================",
    "Initializing neural core...",
    "Loading memory banks...",
    "Scanning local models...",
    "Calibrating personality matrix...",
    "Connecting to reality...",
    "",
    "> SYSTEM READY",
    "> Hello, friend."
)

@Composable
fun BootScreen(onBootComplete: () -> Unit) {
    var visibleLines by remember { mutableIntStateOf(0) }
    var cursorVisible by remember { mutableStateOf(true) }

    // Blink cursor
    LaunchedEffect(Unit) {
        while (true) {
            delay(500)
            cursorVisible = !cursorVisible
        }
    }

    // Reveal lines one by one
    LaunchedEffect(Unit) {
        for (i in bootLines.indices) {
            delay(if (i < 2) 200L else 400L)
            visibleLines = i + 1
        }
        delay(1000)
        onBootComplete()
    }

    Box(
        Modifier
            .fillMaxSize()
            .background(RetroDark)
    ) {
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(24.dp),
            verticalArrangement = Arrangement.Center
        ) {
            bootLines.take(visibleLines).forEachIndexed { index, line ->
                val color = when {
                    index == 0 -> RetroGreen
                    line.startsWith(">") -> RetroAmber
                    line.startsWith("=") -> RetroGreen
                    else -> RetroCyan
                }
                Text(
                    text = line,
                    style = RetroTypography.labelMedium,
                    color = color,
                    textAlign = TextAlign.Start
                )
                Spacer(Modifier.height(4.dp))
            }

            if (visibleLines > 0 && visibleLines <= bootLines.size) {
                Text(
                    text = if (cursorVisible) "█" else " ",
                    style = RetroTypography.labelMedium,
                    color = RetroGreen
                )
            }
        }

        ScanlineOverlay(alpha = 0.05f)
    }
}
```

- [ ] **Step 2: Add boot route to NavHost**

Add `"boot"` as startDestination, navigate to `Screen.Chat.route` on boot complete:
```kotlin
composable("boot") {
    BootScreen(onBootComplete = {
        navController.navigate(Screen.Chat.route) {
            popUpTo("boot") { inclusive = true }
        }
    })
}
```
Change `startDestination = "boot"`. Hide GameHudNavBar when on boot screen.

- [ ] **Step 3: Build and verify**

Run: `cd android && ./gradlew assembleDebug`
Expected: BUILD SUCCESSFUL

- [ ] **Step 4: Commit**

```bash
git add android/app/src/main/kotlin/com/heyhomie/app/ui/
git commit -m "feat(android): add retro boot screen with typewriter ASCII startup"
```

---

### Task 6: Chat Screen — Terminal Style with Typewriter Animation

**Files:**
- Create: `android/app/src/main/kotlin/com/heyhomie/app/ui/screens/ChatScreen.kt` (replace placeholder)
- Create: `android/app/src/main/kotlin/com/heyhomie/app/ui/components/TypewriterText.kt`
- Create: `android/app/src/main/kotlin/com/heyhomie/app/ui/components/ChatBubble.kt`
- Create: `android/app/src/main/kotlin/com/heyhomie/app/ui/components/RetroTextField.kt`
- Create: `android/app/src/main/kotlin/com/heyhomie/app/core/model/ChatMessage.kt`
- Create: `android/app/src/main/kotlin/com/heyhomie/app/ui/viewmodel/ChatViewModel.kt`
- Test: `android/app/src/test/kotlin/com/heyhomie/app/ui/viewmodel/ChatViewModelTest.kt`

- [ ] **Step 1: Write ChatViewModel test**

```kotlin
package com.heyhomie.app.ui.viewmodel

import com.heyhomie.app.core.model.ChatMessage
import com.heyhomie.app.core.model.MessageRole
import org.junit.Assert.*
import org.junit.Test

class ChatViewModelTest {
    @Test
    fun `sending message adds user message to list`() {
        val vm = ChatViewModel()
        vm.sendMessage("Hello Homie")
        assertEquals(1, vm.messages.value.count { it.role == MessageRole.USER })
        assertEquals("Hello Homie", vm.messages.value.first { it.role == MessageRole.USER }.text)
    }

    @Test
    fun `sending message triggers assistant response`() {
        val vm = ChatViewModel()
        vm.sendMessage("hi")
        // Should have at least user message; assistant will be added async
        assertTrue(vm.messages.value.isNotEmpty())
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd android && ./gradlew testDebugUnitTest --tests "*.ChatViewModelTest"`
Expected: FAIL

- [ ] **Step 3: Create ChatMessage model**

```kotlin
package com.heyhomie.app.core.model

import java.util.UUID

enum class MessageRole { USER, ASSISTANT, SYSTEM }

data class ChatMessage(
    val id: String = UUID.randomUUID().toString(),
    val role: MessageRole,
    val text: String,
    val timestamp: Long = System.currentTimeMillis(),
    val isStreaming: Boolean = false
)
```

- [ ] **Step 4: Create ChatViewModel**

```kotlin
package com.heyhomie.app.ui.viewmodel

import androidx.lifecycle.ViewModel
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import com.heyhomie.app.core.model.ChatMessage
import com.heyhomie.app.core.model.MessageRole

class ChatViewModel : ViewModel() {
    private val _messages = MutableStateFlow<List<ChatMessage>>(emptyList())
    val messages: StateFlow<List<ChatMessage>> = _messages.asStateFlow()

    private val _inputText = MutableStateFlow("")
    val inputText: StateFlow<String> = _inputText.asStateFlow()

    private val _isGenerating = MutableStateFlow(false)
    val isGenerating: StateFlow<Boolean> = _isGenerating.asStateFlow()

    fun updateInput(text: String) {
        _inputText.value = text
    }

    fun sendMessage(text: String) {
        if (text.isBlank()) return
        val userMsg = ChatMessage(role = MessageRole.USER, text = text.trim())
        _messages.value = _messages.value + userMsg
        _inputText.value = ""
        // Placeholder: will be wired to InferenceRouter later
        val placeholder = ChatMessage(
            role = MessageRole.ASSISTANT,
            text = "I'm Homie! Inference not connected yet. Stay tuned, friend.",
            isStreaming = false
        )
        _messages.value = _messages.value + placeholder
    }
}
```

- [ ] **Step 5: Create TypewriterText.kt**

```kotlin
package com.heyhomie.app.ui.components

import androidx.compose.material3.Text
import androidx.compose.runtime.*
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.TextStyle
import com.heyhomie.app.ui.theme.RetroGreen
import com.heyhomie.app.ui.theme.RetroTypography
import kotlinx.coroutines.delay

@Composable
fun TypewriterText(
    fullText: String,
    modifier: Modifier = Modifier,
    style: TextStyle = RetroTypography.bodyMedium,
    color: Color = RetroGreen,
    charDelayMs: Long = 20L,
    onComplete: () -> Unit = {}
) {
    var visibleCount by remember(fullText) { mutableIntStateOf(0) }

    LaunchedEffect(fullText) {
        visibleCount = 0
        for (i in fullText.indices) {
            delay(charDelayMs)
            visibleCount = i + 1
        }
        onComplete()
    }

    Text(
        text = fullText.take(visibleCount) + if (visibleCount < fullText.length) "█" else "",
        style = style,
        color = color,
        modifier = modifier
    )
}
```

- [ ] **Step 6: Create ChatBubble.kt**

```kotlin
package com.heyhomie.app.ui.components

import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import com.heyhomie.app.core.model.ChatMessage
import com.heyhomie.app.core.model.MessageRole
import com.heyhomie.app.ui.theme.*

@Composable
fun ChatBubble(
    message: ChatMessage,
    modifier: Modifier = Modifier
) {
    val isUser = message.role == MessageRole.USER
    val alignment = if (isUser) Alignment.CenterEnd else Alignment.CenterStart
    val bgColor = if (isUser) RetroDimGreen else RetroDarkCard
    val textColor = if (isUser) RetroWhite else RetroGreen
    val prefix = if (isUser) "> " else "HOMIE> "

    Box(
        modifier = modifier.fillMaxWidth(),
        contentAlignment = alignment
    ) {
        Column(
            modifier = Modifier
                .widthIn(max = 300.dp)
                .let { if (!isUser) it.crtGlow(RetroGreen, 8f) else it }
                .pixelBorder(color = if (isUser) RetroAmber else RetroGreen, width = 1f)
                .background(bgColor)
                .padding(10.dp)
        ) {
            if (!isUser) {
                Text("🤖", style = RetroTypography.labelMedium)
                Spacer(Modifier.height(4.dp))
            }

            if (!isUser && message.isStreaming) {
                TypewriterText(
                    fullText = message.text,
                    color = textColor
                )
            } else {
                Text(
                    text = "$prefix${message.text}",
                    style = RetroTypography.bodyMedium,
                    color = textColor
                )
            }
        }
    }
}
```

- [ ] **Step 7: Create RetroTextField.kt**

```kotlin
package com.heyhomie.app.ui.components

import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.text.BasicTextField
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.SolidColor
import androidx.compose.ui.unit.dp
import com.heyhomie.app.ui.theme.*

@Composable
fun RetroTextField(
    value: String,
    onValueChange: (String) -> Unit,
    onSend: () -> Unit,
    modifier: Modifier = Modifier,
    enabled: Boolean = true
) {
    Row(
        modifier = modifier
            .fillMaxWidth()
            .pixelBorder(color = RetroGreen, width = 1f)
            .background(RetroDark)
            .padding(8.dp),
        verticalAlignment = Alignment.CenterVertically
    ) {
        Text("> ", style = RetroTypography.bodyMedium, color = RetroGreen)
        BasicTextField(
            value = value,
            onValueChange = onValueChange,
            enabled = enabled,
            textStyle = RetroTypography.bodyMedium.copy(color = RetroWhite),
            cursorBrush = SolidColor(RetroGreen),
            modifier = Modifier.weight(1f),
            singleLine = true
        )
        Text(
            text = "[SEND]",
            style = RetroTypography.labelMedium,
            color = if (value.isNotBlank()) RetroAmber else RetroGray,
            modifier = Modifier
                .padding(start = 8.dp)
                .then(
                    if (value.isNotBlank()) Modifier.pixelBorder(RetroAmber, width = 1f)
                    else Modifier
                )
                .padding(4.dp)
        )
    }
}
```

- [ ] **Step 8: Replace ChatScreen placeholder**

```kotlin
package com.heyhomie.app.ui.screens

import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.lazy.rememberLazyListState
import androidx.compose.runtime.*
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import androidx.lifecycle.viewmodel.compose.viewModel
import com.heyhomie.app.ui.components.*
import com.heyhomie.app.ui.theme.RetroDark
import com.heyhomie.app.ui.viewmodel.ChatViewModel

@Composable
fun ChatScreen(viewModel: ChatViewModel = viewModel()) {
    val messages by viewModel.messages.collectAsState()
    val inputText by viewModel.inputText.collectAsState()
    val isGenerating by viewModel.isGenerating.collectAsState()
    val listState = rememberLazyListState()

    LaunchedEffect(messages.size) {
        if (messages.isNotEmpty()) {
            listState.animateScrollToItem(messages.lastIndex)
        }
    }

    Box(Modifier.fillMaxSize().background(RetroDark)) {
        Column(Modifier.fillMaxSize()) {
            // Chat messages
            LazyColumn(
                state = listState,
                modifier = Modifier.weight(1f).padding(horizontal = 12.dp),
                verticalArrangement = Arrangement.spacedBy(8.dp),
                contentPadding = PaddingValues(vertical = 8.dp)
            ) {
                items(messages, key = { it.id }) { message ->
                    ChatBubble(message = message)
                }
            }

            // Input
            RetroTextField(
                value = inputText,
                onValueChange = { viewModel.updateInput(it) },
                onSend = { viewModel.sendMessage(inputText) },
                enabled = !isGenerating,
                modifier = Modifier.padding(8.dp)
            )
        }

        ScanlineOverlay(alpha = 0.03f)
    }
}
```

- [ ] **Step 9: Run tests**

Run: `cd android && ./gradlew testDebugUnitTest --tests "*.ChatViewModelTest"`
Expected: PASS

- [ ] **Step 10: Commit**

```bash
git add android/app/src/main/kotlin/com/heyhomie/app/
git add android/app/src/test/
git commit -m "feat(android): add terminal-style chat UI with typewriter animation and retro input"
```

---

## Chunk 3: Data Layer & Inference

### Task 7: Room Database — Conversations & Memory

**Files:**
- Create: `android/app/src/main/kotlin/com/heyhomie/app/core/data/HomieDatabase.kt`
- Create: `android/app/src/main/kotlin/com/heyhomie/app/core/data/entity/MessageEntity.kt`
- Create: `android/app/src/main/kotlin/com/heyhomie/app/core/data/entity/MemoryEntity.kt`
- Create: `android/app/src/main/kotlin/com/heyhomie/app/core/data/dao/MessageDao.kt`
- Create: `android/app/src/main/kotlin/com/heyhomie/app/core/data/dao/MemoryDao.kt`
- Create: `android/app/src/main/kotlin/com/heyhomie/app/core/data/di/DatabaseModule.kt`
- Test: `android/app/src/test/kotlin/com/heyhomie/app/core/data/entity/EntityTest.kt`

- [ ] **Step 1: Write entity test**

```kotlin
package com.heyhomie.app.core.data.entity

import org.junit.Assert.*
import org.junit.Test

class EntityTest {
    @Test
    fun `message entity defaults timestamp to current time`() {
        val entity = MessageEntity(role = "user", text = "hello", conversationId = "conv1")
        assertTrue(entity.timestamp > 0)
    }

    @Test
    fun `memory entity stores type and content`() {
        val entity = MemoryEntity(type = "episodic", content = "User likes coffee", deviceId = "dev1")
        assertEquals("episodic", entity.type)
    }
}
```

- [ ] **Step 2: Run test — fails**

- [ ] **Step 3: Create entities**

`MessageEntity.kt`:
```kotlin
package com.heyhomie.app.core.data.entity

import androidx.room.Entity
import androidx.room.PrimaryKey
import java.util.UUID

@Entity(tableName = "messages")
data class MessageEntity(
    @PrimaryKey val id: String = UUID.randomUUID().toString(),
    val conversationId: String,
    val role: String,
    val text: String,
    val timestamp: Long = System.currentTimeMillis(),
    val deviceId: String = ""
)
```

`MemoryEntity.kt`:
```kotlin
package com.heyhomie.app.core.data.entity

import androidx.room.Entity
import androidx.room.PrimaryKey
import java.util.UUID

@Entity(tableName = "memories")
data class MemoryEntity(
    @PrimaryKey val id: String = UUID.randomUUID().toString(),
    val type: String, // "episodic", "semantic", "working"
    val content: String,
    val contentHash: String = "",
    val deviceId: String,
    val timestamp: Long = System.currentTimeMillis(),
    val tombstone: Boolean = false,
    val lamportClock: Long = 0
)
```

- [ ] **Step 4: Create DAOs**

`MessageDao.kt`:
```kotlin
package com.heyhomie.app.core.data.dao

import androidx.room.*
import com.heyhomie.app.core.data.entity.MessageEntity
import kotlinx.coroutines.flow.Flow

@Dao
interface MessageDao {
    @Query("SELECT * FROM messages WHERE conversationId = :convId ORDER BY timestamp ASC")
    fun getMessages(convId: String): Flow<List<MessageEntity>>

    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun insert(message: MessageEntity)

    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun insertAll(messages: List<MessageEntity>)

    @Query("SELECT DISTINCT conversationId FROM messages ORDER BY timestamp DESC")
    fun getConversationIds(): Flow<List<String>>
}
```

`MemoryDao.kt`:
```kotlin
package com.heyhomie.app.core.data.dao

import androidx.room.*
import com.heyhomie.app.core.data.entity.MemoryEntity
import kotlinx.coroutines.flow.Flow

@Dao
interface MemoryDao {
    @Query("SELECT * FROM memories WHERE type = :type AND tombstone = 0 ORDER BY timestamp DESC")
    fun getByType(type: String): Flow<List<MemoryEntity>>

    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun insert(memory: MemoryEntity)

    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun insertAll(memories: List<MemoryEntity>)

    @Query("UPDATE memories SET tombstone = 1 WHERE id = :id")
    suspend fun markTombstone(id: String)

    @Query("SELECT * FROM memories WHERE timestamp > :since AND tombstone = 0")
    suspend fun getNewerThan(since: Long): List<MemoryEntity>
}
```

- [ ] **Step 5: Create Database**

```kotlin
package com.heyhomie.app.core.data

import androidx.room.Database
import androidx.room.RoomDatabase
import com.heyhomie.app.core.data.dao.MessageDao
import com.heyhomie.app.core.data.dao.MemoryDao
import com.heyhomie.app.core.data.entity.MessageEntity
import com.heyhomie.app.core.data.entity.MemoryEntity

@Database(
    entities = [MessageEntity::class, MemoryEntity::class],
    version = 1,
    exportSchema = false
)
abstract class HomieDatabase : RoomDatabase() {
    abstract fun messageDao(): MessageDao
    abstract fun memoryDao(): MemoryDao
}
```

- [ ] **Step 6: Create DI module**

```kotlin
package com.heyhomie.app.core.data.di

import android.content.Context
import androidx.room.Room
import com.heyhomie.app.core.data.HomieDatabase
import com.heyhomie.app.core.data.dao.MessageDao
import com.heyhomie.app.core.data.dao.MemoryDao
import dagger.Module
import dagger.Provides
import dagger.hilt.InstallIn
import dagger.hilt.android.qualifiers.ApplicationContext
import dagger.hilt.components.SingletonComponent
import javax.inject.Singleton

@Module
@InstallIn(SingletonComponent::class)
object DatabaseModule {
    @Provides @Singleton
    fun provideDatabase(@ApplicationContext ctx: Context): HomieDatabase =
        Room.databaseBuilder(ctx, HomieDatabase::class.java, "homie.db").build()

    @Provides fun provideMessageDao(db: HomieDatabase): MessageDao = db.messageDao()
    @Provides fun provideMemoryDao(db: HomieDatabase): MemoryDao = db.memoryDao()
}
```

- [ ] **Step 7: Run tests — passes**

- [ ] **Step 8: Commit**

```bash
git add android/app/src/main/kotlin/com/heyhomie/app/core/data/
git add android/app/src/test/
git commit -m "feat(android): add Room database with message and memory entities"
```

---

### Task 8: Qubrid Cloud Client

**Files:**
- Create: `android/app/src/main/kotlin/com/heyhomie/app/core/inference/QubridClient.kt`
- Test: `android/app/src/test/kotlin/com/heyhomie/app/core/inference/QubridClientTest.kt`

- [ ] **Step 1: Write test**

```kotlin
package com.heyhomie.app.core.inference

import org.junit.Assert.*
import org.junit.Test

class QubridClientTest {
    @Test
    fun `default model is Qwen3 5 Flash`() {
        val client = QubridClient(apiKey = "test-key")
        assertEquals("Qwen/Qwen3.5-Flash", client.model)
    }

    @Test
    fun `base url defaults to qubrid platform`() {
        val client = QubridClient(apiKey = "test-key")
        assertEquals("https://platform.qubrid.com/v1", client.baseUrl)
    }

    @Test
    fun `isAvailable returns false without api key`() {
        val client = QubridClient(apiKey = "")
        assertFalse(client.isAvailable)
    }

    @Test
    fun `isAvailable returns true with api key`() {
        val client = QubridClient(apiKey = "real-key")
        assertTrue(client.isAvailable)
    }
}
```

- [ ] **Step 2: Run test — fails**

- [ ] **Step 3: Implement QubridClient**

```kotlin
package com.heyhomie.app.core.inference

import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import okhttp3.*
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.RequestBody.Companion.toRequestBody
import org.json.JSONArray
import org.json.JSONObject

class QubridClient(
    private val apiKey: String,
    val baseUrl: String = "https://platform.qubrid.com/v1",
    val model: String = "Qwen/Qwen3.5-Flash",
    private val timeout: Long = 60_000
) {
    private val client = OkHttpClient.Builder()
        .connectTimeout(timeout, java.util.concurrent.TimeUnit.MILLISECONDS)
        .readTimeout(timeout, java.util.concurrent.TimeUnit.MILLISECONDS)
        .build()

    val isAvailable: Boolean get() = apiKey.isNotBlank()

    suspend fun generate(prompt: String, systemPrompt: String? = null): String =
        withContext(Dispatchers.IO) {
            val messages = JSONArray().apply {
                systemPrompt?.let {
                    put(JSONObject().put("role", "system").put("content", it))
                }
                put(JSONObject().put("role", "user").put("content", prompt))
            }
            val body = JSONObject().apply {
                put("model", model)
                put("messages", messages)
                put("max_tokens", 2048)
            }
            val request = Request.Builder()
                .url("$baseUrl/chat/completions")
                .addHeader("Authorization", "Bearer $apiKey")
                .addHeader("Content-Type", "application/json")
                .post(body.toString().toRequestBody("application/json".toMediaType()))
                .build()

            val response = client.newCall(request).execute()
            if (!response.isSuccessful) {
                throw RuntimeException("Qubrid API error: ${response.code} ${response.message}")
            }
            val json = JSONObject(response.body?.string() ?: "")
            json.getJSONArray("choices")
                .getJSONObject(0)
                .getJSONObject("message")
                .getString("content")
        }
}
```

- [ ] **Step 4: Run tests — passes**

- [ ] **Step 5: Commit**

```bash
git add android/app/src/main/kotlin/com/heyhomie/app/core/inference/
git add android/app/src/test/
git commit -m "feat(android): add Qubrid cloud inference client"
```

---

### Task 9: Inference Router

**Files:**
- Create: `android/app/src/main/kotlin/com/heyhomie/app/core/inference/InferenceRouter.kt`
- Create: `android/app/src/main/kotlin/com/heyhomie/app/core/inference/InferenceSource.kt`
- Test: `android/app/src/test/kotlin/com/heyhomie/app/core/inference/InferenceRouterTest.kt`

- [ ] **Step 1: Write test**

```kotlin
package com.heyhomie.app.core.inference

import kotlinx.coroutines.test.runTest
import org.junit.Assert.*
import org.junit.Test

class InferenceRouterTest {
    @Test
    fun `active source returns Qubrid when no local model`() = runTest {
        val qubrid = QubridClient(apiKey = "test")
        val router = InferenceRouter(localBridge = null, qubridClient = qubrid)
        assertEquals("Homie Intelligence (Cloud)", router.activeSourceName)
    }

    @Test
    fun `fallback banner shown when using cloud`() = runTest {
        val qubrid = QubridClient(apiKey = "test")
        val router = InferenceRouter(localBridge = null, qubridClient = qubrid)
        assertEquals(
            "No local model found! Using Homie's intelligence until local model is setup!",
            router.fallbackBanner
        )
    }

    @Test
    fun `fallback banner null when local model available`() {
        val router = InferenceRouter(localBridge = FakeLocalBridge(), qubridClient = null)
        assertNull(router.fallbackBanner)
    }

    @Test
    fun `no sources throws when generating`() = runTest {
        val router = InferenceRouter(localBridge = null, qubridClient = null)
        try {
            router.generate("hello")
            fail("Should throw")
        } catch (e: IllegalStateException) {
            assertTrue(e.message!!.contains("No inference source"))
        }
    }
}

private class FakeLocalBridge : LocalInferenceBridge {
    override val isAvailable: Boolean = true
    override val modelName: String = "test-model"
    override suspend fun generate(prompt: String, systemPrompt: String?): String = "local response"
}
```

- [ ] **Step 2: Run test — fails**

- [ ] **Step 3: Create InferenceSource.kt**

```kotlin
package com.heyhomie.app.core.inference

interface LocalInferenceBridge {
    val isAvailable: Boolean
    val modelName: String
    suspend fun generate(prompt: String, systemPrompt: String? = null): String
}

enum class InferenceSourceType { LOCAL, LAN, QUBRID }
```

- [ ] **Step 4: Create InferenceRouter.kt**

```kotlin
package com.heyhomie.app.core.inference

class InferenceRouter(
    private val localBridge: LocalInferenceBridge? = null,
    private val qubridClient: QubridClient? = null
    // LAN bridge added in Task 14
) {
    companion object {
        private const val FALLBACK_BANNER =
            "No local model found! Using Homie's intelligence until local model is setup!"
    }

    val activeSourceName: String
        get() = when {
            localBridge?.isAvailable == true -> "Local (${localBridge.modelName})"
            qubridClient?.isAvailable == true -> "Homie Intelligence (Cloud)"
            else -> "None"
        }

    val activeSourceType: InferenceSourceType?
        get() = when {
            localBridge?.isAvailable == true -> InferenceSourceType.LOCAL
            qubridClient?.isAvailable == true -> InferenceSourceType.QUBRID
            else -> null
        }

    val fallbackBanner: String?
        get() = when {
            localBridge?.isAvailable == true -> null
            qubridClient?.isAvailable == true -> FALLBACK_BANNER
            else -> null
        }

    suspend fun generate(prompt: String, systemPrompt: String? = null): String {
        // Priority: local → LAN → qubrid
        if (localBridge?.isAvailable == true) {
            return try {
                localBridge.generate(prompt, systemPrompt)
            } catch (e: Exception) {
                fallbackGenerate(prompt, systemPrompt)
            }
        }
        return fallbackGenerate(prompt, systemPrompt)
    }

    private suspend fun fallbackGenerate(prompt: String, systemPrompt: String?): String {
        // LAN will be inserted here in Task 14
        if (qubridClient?.isAvailable == true) {
            return qubridClient.generate(prompt, systemPrompt)
        }
        throw IllegalStateException("No inference source available")
    }
}
```

- [ ] **Step 5: Run tests — passes**

- [ ] **Step 6: Commit**

```bash
git add android/app/src/main/kotlin/com/heyhomie/app/core/inference/
git add android/app/src/test/
git commit -m "feat(android): add inference router with local→cloud fallback chain"
```

---

## Chunk 4: Phone Analysis & Device Profiling

### Task 10: Device Profiler

**Files:**
- Create: `android/app/src/main/kotlin/com/heyhomie/app/phone/DeviceProfiler.kt`
- Create: `android/app/src/main/kotlin/com/heyhomie/app/phone/DeviceProfile.kt`
- Test: `android/app/src/test/kotlin/com/heyhomie/app/phone/DeviceProfileTest.kt`

- [ ] **Step 1: Write test**

```kotlin
package com.heyhomie.app.phone

import org.junit.Assert.*
import org.junit.Test

class DeviceProfileTest {
    @Test
    fun `capability score calculated from RAM and cores`() {
        val profile = DeviceProfile(
            cpuCores = 8, cpuArch = "arm64-v8a",
            totalRamMb = 8192, availableRamMb = 4096,
            totalStorageMb = 128000, freeStorageMb = 64000,
            batteryLevel = 85, isCharging = false,
            screenDensity = 440, refreshRate = 60f,
            gpuRenderer = "Adreno 730", supportsVulkan = true
        )
        // 8GB RAM + 8 cores + Vulkan = high score
        assertTrue(profile.capabilityScore >= 70)
        assertEquals("7B Q4", profile.recommendedModelSize)
    }

    @Test
    fun `low spec device gets small model recommendation`() {
        val profile = DeviceProfile(
            cpuCores = 4, cpuArch = "armeabi-v7a",
            totalRamMb = 3072, availableRamMb = 1024,
            totalStorageMb = 32000, freeStorageMb = 8000,
            batteryLevel = 50, isCharging = false,
            screenDensity = 320, refreshRate = 60f,
            gpuRenderer = "Mali-G52", supportsVulkan = false
        )
        assertTrue(profile.capabilityScore < 50)
        assertEquals("1.5B Q4", profile.recommendedModelSize)
    }
}
```

- [ ] **Step 2: Run test — fails**

- [ ] **Step 3: Create DeviceProfile data class**

```kotlin
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
    /** 0-100 score indicating device capability for local inference */
    val capabilityScore: Int
        get() {
            var score = 0
            // RAM (biggest factor)
            score += when {
                totalRamMb >= 8192 -> 35
                totalRamMb >= 6144 -> 25
                totalRamMb >= 4096 -> 15
                else -> 5
            }
            // CPU cores
            score += when {
                cpuCores >= 8 -> 20
                cpuCores >= 6 -> 15
                cpuCores >= 4 -> 10
                else -> 5
            }
            // Architecture
            score += if (cpuArch.contains("arm64") || cpuArch.contains("v8a")) 15 else 5
            // GPU/Vulkan
            score += if (supportsVulkan) 15 else 5
            // Storage
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
```

- [ ] **Step 4: Create DeviceProfiler (Android context-dependent)**

```kotlin
package com.heyhomie.app.phone

import android.app.ActivityManager
import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.hardware.SensorManager
import android.os.BatteryManager
import android.os.Build
import android.os.Environment
import android.os.StatFs
import android.view.WindowManager
import android.opengl.GLES20
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

        val sensorManager = context.getSystemService(Context.SENSOR_SERVICE) as SensorManager
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
            gpuRenderer = "unknown", // requires GL context
            supportsVulkan = Build.VERSION.SDK_INT >= 24,
            sensors = sensors
        )
    }
}
```

- [ ] **Step 5: Run tests — passes**

- [ ] **Step 6: Commit**

```bash
git add android/app/src/main/kotlin/com/heyhomie/app/phone/
git add android/app/src/test/
git commit -m "feat(android): add device profiler with capability scoring"
```

---

### Task 11: Phone Stats Screen — Retro Game Dashboard

**Files:**
- Modify: `android/app/src/main/kotlin/com/heyhomie/app/ui/screens/PhoneStatsScreen.kt`
- Create: `android/app/src/main/kotlin/com/heyhomie/app/ui/viewmodel/PhoneStatsViewModel.kt`
- Create: `android/app/src/main/kotlin/com/heyhomie/app/ui/components/StatBar.kt`

- [ ] **Step 1: Create StatBar.kt (HP/XP style bar)**

```kotlin
package com.heyhomie.app.ui.components

import androidx.compose.foundation.layout.*
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.unit.dp
import com.heyhomie.app.ui.theme.*

@Composable
fun StatBar(
    label: String,
    value: Float,
    maxValue: Float = 100f,
    color: Color = RetroGreen,
    suffix: String = "",
    modifier: Modifier = Modifier
) {
    Column(modifier = modifier.fillMaxWidth()) {
        Row(
            Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.SpaceBetween
        ) {
            Text(label, style = RetroTypography.labelMedium, color = RetroAmber)
            Text(
                "${value.toInt()}${suffix}",
                style = RetroTypography.labelMedium,
                color = color
            )
        }
        Spacer(Modifier.height(4.dp))
        RetroProgressBar(
            progress = (value / maxValue).coerceIn(0f, 1f),
            color = color
        )
    }
}
```

- [ ] **Step 2: Create PhoneStatsViewModel**

```kotlin
package com.heyhomie.app.ui.viewmodel

import androidx.lifecycle.ViewModel
import com.heyhomie.app.phone.DeviceProfile
import com.heyhomie.app.phone.DeviceProfiler
import dagger.hilt.android.lifecycle.HiltViewModel
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import javax.inject.Inject

@HiltViewModel
class PhoneStatsViewModel @Inject constructor(
    private val profiler: DeviceProfiler
) : ViewModel() {
    private val _profile = MutableStateFlow<DeviceProfile?>(null)
    val profile: StateFlow<DeviceProfile?> = _profile

    init { refresh() }

    fun refresh() {
        _profile.value = profiler.profile()
    }
}
```

- [ ] **Step 3: Replace PhoneStatsScreen**

```kotlin
package com.heyhomie.app.ui.screens

import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Text
import androidx.compose.runtime.*
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import androidx.hilt.navigation.compose.hiltViewModel
import com.heyhomie.app.ui.components.*
import com.heyhomie.app.ui.theme.*
import com.heyhomie.app.ui.viewmodel.PhoneStatsViewModel

@Composable
fun PhoneStatsScreen(viewModel: PhoneStatsViewModel = hiltViewModel()) {
    val profile by viewModel.profile.collectAsState()

    Column(
        modifier = Modifier
            .fillMaxSize()
            .background(RetroDark)
            .verticalScroll(rememberScrollState())
            .padding(16.dp),
        verticalArrangement = Arrangement.spacedBy(16.dp)
    ) {
        Text("DEVICE STATS", style = RetroTypography.headlineMedium)

        profile?.let { p ->
            // Capability score
            RetroCard {
                Column {
                    Text("CAPABILITY SCORE", style = RetroTypography.titleMedium)
                    Spacer(Modifier.height(8.dp))
                    Text(
                        "${p.capabilityScore}/100",
                        style = RetroTypography.displayLarge,
                        color = when {
                            p.capabilityScore >= 70 -> RetroGreen
                            p.capabilityScore >= 50 -> RetroAmber
                            else -> RetroRed
                        }
                    )
                    Text(
                        "RECOMMENDED: ${p.recommendedModelSize}",
                        style = RetroTypography.labelMedium,
                        color = RetroCyan
                    )
                }
            }

            // Battery — HP bar
            RetroCard {
                StatBar(
                    label = "♥ HP (BATTERY)",
                    value = p.batteryLevel.toFloat(),
                    color = when {
                        p.batteryLevel > 50 -> RetroGreen
                        p.batteryLevel > 20 -> RetroAmber
                        else -> RetroRed
                    },
                    suffix = "%"
                )
            }

            // Storage — XP bar
            RetroCard {
                val usedPct = ((p.totalStorageMb - p.freeStorageMb).toFloat() / p.totalStorageMb * 100)
                StatBar(
                    label = "★ XP (STORAGE)",
                    value = usedPct,
                    color = RetroCyan,
                    suffix = "% used"
                )
            }

            // RAM
            RetroCard {
                StatBar(
                    label = "◆ MP (RAM)",
                    value = p.availableRamMb.toFloat(),
                    maxValue = p.totalRamMb.toFloat(),
                    color = RetroAmber,
                    suffix = " MB free"
                )
            }

            // CPU info
            RetroCard {
                Column {
                    Text("CPU", style = RetroTypography.titleMedium)
                    Text("Cores: ${p.cpuCores}", style = RetroTypography.bodyMedium)
                    Text("Arch: ${p.cpuArch}", style = RetroTypography.bodyMedium)
                }
            }

            // GPU
            RetroCard {
                Column {
                    Text("GPU", style = RetroTypography.titleMedium)
                    Text("Renderer: ${p.gpuRenderer}", style = RetroTypography.bodyMedium)
                    Text("Vulkan: ${if (p.supportsVulkan) "YES" else "NO"}", style = RetroTypography.bodyMedium, color = if (p.supportsVulkan) RetroGreen else RetroRed)
                }
            }

            // Charging status badge
            if (p.isCharging) {
                Text("⚡ CHARGING", style = RetroTypography.labelMedium, color = RetroAmber)
            }
        } ?: Text("Scanning device...", style = RetroTypography.bodyMedium, color = RetroCyan)

        ScanlineOverlay(alpha = 0.03f)
    }
}
```

- [ ] **Step 4: Build and verify**

Run: `cd android && ./gradlew assembleDebug`
Expected: BUILD SUCCESSFUL

- [ ] **Step 5: Commit**

```bash
git add android/app/src/main/kotlin/com/heyhomie/app/ui/
git commit -m "feat(android): add retro game-style phone stats dashboard with HP/XP bars"
```

---

### Task 12: Usage Analyzer & Message Reader

**Files:**
- Create: `android/app/src/main/kotlin/com/heyhomie/app/phone/UsageAnalyzer.kt`
- Create: `android/app/src/main/kotlin/com/heyhomie/app/phone/MessageReader.kt`
- Test: `android/app/src/test/kotlin/com/heyhomie/app/phone/UsageAnalyzerTest.kt`

- [ ] **Step 1: Write test**

```kotlin
package com.heyhomie.app.phone

import org.junit.Assert.*
import org.junit.Test

class UsageAnalyzerTest {
    @Test
    fun `app usage entry has required fields`() {
        val entry = AppUsageEntry(
            packageName = "com.example",
            appName = "Example",
            totalTimeMs = 3600000,
            lastUsed = System.currentTimeMillis(),
            launchCount = 42
        )
        assertEquals("com.example", entry.packageName)
        assertEquals(42, entry.launchCount)
        assertEquals(60, entry.totalTimeMinutes)
    }
}
```

- [ ] **Step 2: Run test — fails**

- [ ] **Step 3: Create UsageAnalyzer**

```kotlin
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
                    launchCount = 0 // not available in older APIs
                )
            }
            .sortedByDescending { it.totalTimeMs }
    }

    fun getTotalScreenTimeMinutes(): Int = getDailyUsage().sumOf { it.totalTimeMinutes }
}
```

- [ ] **Step 4: Create MessageReader**

```kotlin
package com.heyhomie.app.phone

import android.content.Context
import android.net.Uri
import dagger.hilt.android.qualifiers.ApplicationContext
import javax.inject.Inject
import javax.inject.Singleton

data class SmsMessage(
    val address: String,
    val body: String,
    val date: Long,
    val type: Int // 1=received, 2=sent
) {
    val isIncoming: Boolean get() = type == 1
}

@Singleton
class MessageReader @Inject constructor(
    @ApplicationContext private val context: Context
) {
    fun getRecentMessages(limit: Int = 50): List<SmsMessage> {
        val messages = mutableListOf<SmsMessage>()
        val cursor = context.contentResolver.query(
            Uri.parse("content://sms/inbox"),
            arrayOf("address", "body", "date", "type"),
            null, null, "date DESC"
        ) ?: return emptyList()

        cursor.use {
            var count = 0
            while (it.moveToNext() && count < limit) {
                messages.add(
                    SmsMessage(
                        address = it.getString(0) ?: "",
                        body = it.getString(1) ?: "",
                        date = it.getLong(2),
                        type = it.getInt(3)
                    )
                )
                count++
            }
        }
        return messages
    }

    fun searchMessages(query: String): List<SmsMessage> {
        val cursor = context.contentResolver.query(
            Uri.parse("content://sms"),
            arrayOf("address", "body", "date", "type"),
            "body LIKE ?", arrayOf("%$query%"), "date DESC"
        ) ?: return emptyList()

        val results = mutableListOf<SmsMessage>()
        cursor.use {
            while (it.moveToNext()) {
                results.add(
                    SmsMessage(
                        address = it.getString(0) ?: "",
                        body = it.getString(1) ?: "",
                        date = it.getLong(2),
                        type = it.getInt(3)
                    )
                )
            }
        }
        return results
    }
}
```

- [ ] **Step 5: Run tests — passes**

- [ ] **Step 6: Commit**

```bash
git add android/app/src/main/kotlin/com/heyhomie/app/phone/
git add android/app/src/test/
git commit -m "feat(android): add usage analyzer and SMS message reader"
```

---

### Task 13: Notification Listener Service

**Files:**
- Create: `android/app/src/main/kotlin/com/heyhomie/app/notifications/HomieNotificationListener.kt`
- Create: `android/app/src/main/kotlin/com/heyhomie/app/notifications/NotificationStore.kt`
- Modify: `android/app/src/main/AndroidManifest.xml`

- [ ] **Step 1: Create NotificationStore**

```kotlin
package com.heyhomie.app.notifications

import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import javax.inject.Inject
import javax.inject.Singleton

data class CapturedNotification(
    val packageName: String,
    val appName: String,
    val title: String,
    val text: String,
    val timestamp: Long,
    val category: String? = null,
    val priority: Int = 0
)

@Singleton
class NotificationStore @Inject constructor() {
    private val _notifications = MutableStateFlow<List<CapturedNotification>>(emptyList())
    val notifications: StateFlow<List<CapturedNotification>> = _notifications

    fun add(notification: CapturedNotification) {
        _notifications.value = (_notifications.value + notification).takeLast(200)
    }

    fun getUnread(since: Long): List<CapturedNotification> =
        _notifications.value.filter { it.timestamp > since }

    fun summarize(): String {
        val recent = getUnread(System.currentTimeMillis() - 3600_000)
        if (recent.isEmpty()) return "No new notifications in the last hour."
        val byApp = recent.groupBy { it.appName }
        val parts = byApp.map { (app, notifs) -> "$app: ${notifs.size}" }
        return "You got ${recent.size} notifications in the last hour — ${parts.joinToString(", ")}"
    }

    fun clear() { _notifications.value = emptyList() }
}
```

- [ ] **Step 2: Create HomieNotificationListener**

```kotlin
package com.heyhomie.app.notifications

import android.service.notification.NotificationListenerService
import android.service.notification.StatusBarNotification

class HomieNotificationListener : NotificationListenerService() {

    companion object {
        var store: NotificationStore? = null
    }

    override fun onNotificationPosted(sbn: StatusBarNotification) {
        val extras = sbn.notification.extras
        val captured = CapturedNotification(
            packageName = sbn.packageName,
            appName = extras.getString("android.title.big")
                ?: sbn.packageName.substringAfterLast('.'),
            title = extras.getCharSequence("android.title")?.toString() ?: "",
            text = extras.getCharSequence("android.text")?.toString() ?: "",
            timestamp = sbn.postTime,
            category = sbn.notification.category,
            priority = sbn.notification.priority
        )
        store?.add(captured)
    }

    override fun onNotificationRemoved(sbn: StatusBarNotification) {
        // Could track dismissed notifications if needed
    }
}
```

- [ ] **Step 3: Register in AndroidManifest.xml**

Add inside `<application>`:
```xml
<service
    android:name=".notifications.HomieNotificationListener"
    android:permission="android.permission.BIND_NOTIFICATION_LISTENER_SERVICE"
    android:exported="false">
    <intent-filter>
        <action android:name="android.service.notification.NotificationListenerService" />
    </intent-filter>
</service>
```

- [ ] **Step 4: Build and verify**

Run: `cd android && ./gradlew assembleDebug`
Expected: BUILD SUCCESSFUL

- [ ] **Step 5: Commit**

```bash
git add android/app/src/main/kotlin/com/heyhomie/app/notifications/
git add android/app/src/main/AndroidManifest.xml
git commit -m "feat(android): add notification listener service with capture and summarization"
```

---

## Chunk 5: LAN Sync & Settings

### Task 14: LAN Discovery Client

**Files:**
- Create: `android/app/src/main/kotlin/com/heyhomie/app/network/LanDiscovery.kt`
- Create: `android/app/src/main/kotlin/com/heyhomie/app/network/PeerDevice.kt`
- Test: `android/app/src/test/kotlin/com/heyhomie/app/network/PeerDeviceTest.kt`

- [ ] **Step 1: Write test**

```kotlin
package com.heyhomie.app.network

import org.junit.Assert.*
import org.junit.Test

class PeerDeviceTest {
    @Test
    fun `peer device stores name and address`() {
        val peer = PeerDevice(
            name = "Desktop-Homie",
            host = "192.168.1.100",
            port = 8765,
            deviceId = "abc123"
        )
        assertEquals("Desktop-Homie", peer.name)
        assertEquals(8765, peer.port)
    }

    @Test
    fun `websocket url constructed correctly`() {
        val peer = PeerDevice("D", "192.168.1.5", 8765, "x")
        assertEquals("ws://192.168.1.5:8765", peer.wsUrl)
    }
}
```

- [ ] **Step 2: Run test — fails**

- [ ] **Step 3: Create PeerDevice**

```kotlin
package com.heyhomie.app.network

data class PeerDevice(
    val name: String,
    val host: String,
    val port: Int,
    val deviceId: String
) {
    val wsUrl: String get() = "ws://$host:$port"
}
```

- [ ] **Step 4: Create LanDiscovery (NSD-based)**

```kotlin
package com.heyhomie.app.network

import android.content.Context
import android.net.nsd.NsdManager
import android.net.nsd.NsdServiceInfo
import dagger.hilt.android.qualifiers.ApplicationContext
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import javax.inject.Inject
import javax.inject.Singleton

@Singleton
class LanDiscovery @Inject constructor(
    @ApplicationContext private val context: Context
) {
    companion object {
        const val SERVICE_TYPE = "_homie._tcp."
    }

    private val _peers = MutableStateFlow<List<PeerDevice>>(emptyList())
    val peers: StateFlow<List<PeerDevice>> = _peers

    private var nsdManager: NsdManager? = null
    private var discoveryListener: NsdManager.DiscoveryListener? = null

    fun startDiscovery() {
        nsdManager = context.getSystemService(Context.NSD_SERVICE) as NsdManager

        discoveryListener = object : NsdManager.DiscoveryListener {
            override fun onDiscoveryStarted(serviceType: String) {}
            override fun onDiscoveryStopped(serviceType: String) {}
            override fun onStartDiscoveryFailed(serviceType: String, errorCode: Int) {}
            override fun onStopDiscoveryFailed(serviceType: String, errorCode: Int) {}

            override fun onServiceFound(serviceInfo: NsdServiceInfo) {
                nsdManager?.resolveService(serviceInfo, object : NsdManager.ResolveListener {
                    override fun onResolveFailed(si: NsdServiceInfo, errorCode: Int) {}
                    override fun onServiceResolved(si: NsdServiceInfo) {
                        val peer = PeerDevice(
                            name = si.serviceName,
                            host = si.host.hostAddress ?: return,
                            port = si.port,
                            deviceId = si.attributes["device_id"]
                                ?.let { String(it) } ?: si.serviceName
                        )
                        _peers.value = (_peers.value + peer).distinctBy { it.deviceId }
                    }
                })
            }

            override fun onServiceLost(serviceInfo: NsdServiceInfo) {
                _peers.value = _peers.value.filter { it.name != serviceInfo.serviceName }
            }
        }

        nsdManager?.discoverServices(SERVICE_TYPE, NsdManager.PROTOCOL_DNS_SD, discoveryListener)
    }

    fun stopDiscovery() {
        discoveryListener?.let { nsdManager?.stopServiceDiscovery(it) }
        discoveryListener = null
    }
}
```

- [ ] **Step 5: Run tests — passes**

- [ ] **Step 6: Commit**

```bash
git add android/app/src/main/kotlin/com/heyhomie/app/network/
git add android/app/src/test/
git commit -m "feat(android): add LAN discovery client via NSD"
```

---

### Task 15: Sync Client — WebSocket Connection

**Files:**
- Create: `android/app/src/main/kotlin/com/heyhomie/app/network/SyncClient.kt`
- Create: `android/app/src/main/kotlin/com/heyhomie/app/network/Protocol.kt`
- Test: `android/app/src/test/kotlin/com/heyhomie/app/network/ProtocolTest.kt`

- [ ] **Step 1: Write test**

```kotlin
package com.heyhomie.app.network

import org.json.JSONObject
import org.junit.Assert.*
import org.junit.Test

class ProtocolTest {
    @Test
    fun `hello message serializes correctly`() {
        val msg = SyncMessage.hello("android-dev1", "1.0.0", "Android Phone")
        val json = msg.toJson()
        assertEquals("hello", json.getString("type"))
        assertEquals("1.0.0", json.getString("protocol_version"))
        assertEquals("android-dev1", json.getString("device_id"))
    }

    @Test
    fun `inference request serializes correctly`() {
        val msg = SyncMessage.inferenceRequest("Hello Homie", "Be helpful")
        val json = msg.toJson()
        assertEquals("inference_request", json.getString("type"))
        assertEquals("Hello Homie", json.getString("prompt"))
    }

    @Test
    fun `message parsed from json`() {
        val json = JSONObject().apply {
            put("type", "status")
            put("protocol_version", "1.0.0")
            put("device_id", "desktop1")
            put("model_loaded", true)
            put("model_name", "qwen-7b")
        }
        val msg = SyncMessage.fromJson(json)
        assertEquals("status", msg.type)
    }
}
```

- [ ] **Step 2: Run test — fails**

- [ ] **Step 3: Create Protocol.kt**

```kotlin
package com.heyhomie.app.network

import org.json.JSONObject

data class SyncMessage(
    val type: String,
    val protocolVersion: String = "1.0.0",
    val deviceId: String = "",
    val data: JSONObject = JSONObject()
) {
    fun toJson(): JSONObject = JSONObject().apply {
        put("type", type)
        put("protocol_version", protocolVersion)
        put("device_id", deviceId)
        // Merge extra data fields
        data.keys().forEach { key -> put(key, data.get(key)) }
    }

    companion object {
        fun fromJson(json: JSONObject): SyncMessage = SyncMessage(
            type = json.getString("type"),
            protocolVersion = json.optString("protocol_version", "1.0.0"),
            deviceId = json.optString("device_id", ""),
            data = json
        )

        fun hello(deviceId: String, version: String, name: String) = SyncMessage(
            type = "hello", protocolVersion = version, deviceId = deviceId,
            data = JSONObject().put("device_name", name)
        )

        fun inferenceRequest(prompt: String, systemPrompt: String? = null) = SyncMessage(
            type = "inference_request",
            data = JSONObject().apply {
                put("prompt", prompt)
                systemPrompt?.let { put("system_prompt", it) }
            }
        )

        fun memorySyncRequest(since: Long) = SyncMessage(
            type = "memory_sync",
            data = JSONObject().put("since", since)
        )
    }
}
```

- [ ] **Step 4: Create SyncClient**

```kotlin
package com.heyhomie.app.network

import kotlinx.coroutines.*
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import okhttp3.*
import org.json.JSONObject
import java.util.concurrent.TimeUnit
import javax.inject.Inject
import javax.inject.Singleton

@Singleton
class SyncClient @Inject constructor() {

    private val client = OkHttpClient.Builder()
        .readTimeout(0, TimeUnit.MILLISECONDS)
        .build()

    private var webSocket: WebSocket? = null
    private val scope = CoroutineScope(Dispatchers.IO + SupervisorJob())

    private val _connectionState = MutableStateFlow(ConnectionState.DISCONNECTED)
    val connectionState: StateFlow<ConnectionState> = _connectionState

    private val _incomingMessages = MutableStateFlow<SyncMessage?>(null)
    val incomingMessages: StateFlow<SyncMessage?> = _incomingMessages

    var deviceId: String = "android-${System.currentTimeMillis()}"

    fun connect(peer: PeerDevice) {
        _connectionState.value = ConnectionState.CONNECTING
        val request = Request.Builder().url(peer.wsUrl).build()

        webSocket = client.newWebSocket(request, object : WebSocketListener() {
            override fun onOpen(ws: WebSocket, response: Response) {
                _connectionState.value = ConnectionState.CONNECTED
                // Send hello
                val hello = SyncMessage.hello(deviceId, "1.0.0", "Android")
                ws.send(hello.toJson().toString())
            }

            override fun onMessage(ws: WebSocket, text: String) {
                val msg = SyncMessage.fromJson(JSONObject(text))
                _incomingMessages.value = msg
            }

            override fun onFailure(ws: WebSocket, t: Throwable, response: Response?) {
                _connectionState.value = ConnectionState.DISCONNECTED
                // Auto-reconnect after 5s
                scope.launch {
                    delay(5000)
                    connect(peer)
                }
            }

            override fun onClosed(ws: WebSocket, code: Int, reason: String) {
                _connectionState.value = ConnectionState.DISCONNECTED
            }
        })
    }

    fun send(message: SyncMessage) {
        webSocket?.send(message.toJson().toString())
    }

    fun disconnect() {
        webSocket?.close(1000, "User disconnect")
        webSocket = null
        _connectionState.value = ConnectionState.DISCONNECTED
    }
}

enum class ConnectionState { DISCONNECTED, CONNECTING, CONNECTED }
```

- [ ] **Step 5: Run tests — passes**

- [ ] **Step 6: Commit**

```bash
git add android/app/src/main/kotlin/com/heyhomie/app/network/
git add android/app/src/test/
git commit -m "feat(android): add WebSocket sync client and protocol messages"
```

---

### Task 16: Network Screen — Pairing & Status

**Files:**
- Modify: `android/app/src/main/kotlin/com/heyhomie/app/ui/screens/NetworkScreen.kt`
- Create: `android/app/src/main/kotlin/com/heyhomie/app/ui/viewmodel/NetworkViewModel.kt`

- [ ] **Step 1: Create NetworkViewModel**

```kotlin
package com.heyhomie.app.ui.viewmodel

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.heyhomie.app.network.*
import dagger.hilt.android.lifecycle.HiltViewModel
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.launch
import javax.inject.Inject

@HiltViewModel
class NetworkViewModel @Inject constructor(
    private val discovery: LanDiscovery,
    private val syncClient: SyncClient
) : ViewModel() {
    val peers: StateFlow<List<PeerDevice>> = discovery.peers
    val connectionState: StateFlow<ConnectionState> = syncClient.connectionState

    private val _pairingCode = MutableStateFlow("")
    val pairingCode: StateFlow<String> = _pairingCode

    init {
        discovery.startDiscovery()
    }

    fun updatePairingCode(code: String) {
        _pairingCode.value = code.filter { it.isDigit() }.take(6)
    }

    fun connectToPeer(peer: PeerDevice) {
        syncClient.connect(peer)
    }

    fun disconnect() {
        syncClient.disconnect()
    }

    override fun onCleared() {
        discovery.stopDiscovery()
        syncClient.disconnect()
    }
}
```

- [ ] **Step 2: Replace NetworkScreen**

```kotlin
package com.heyhomie.app.ui.screens

import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.material3.Text
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import androidx.hilt.navigation.compose.hiltViewModel
import com.heyhomie.app.network.ConnectionState
import com.heyhomie.app.ui.components.*
import com.heyhomie.app.ui.theme.*
import com.heyhomie.app.ui.viewmodel.NetworkViewModel

@Composable
fun NetworkScreen(viewModel: NetworkViewModel = hiltViewModel()) {
    val peers by viewModel.peers.collectAsState()
    val connectionState by viewModel.connectionState.collectAsState()
    val pairingCode by viewModel.pairingCode.collectAsState()

    Column(
        modifier = Modifier
            .fillMaxSize()
            .background(RetroDark)
            .padding(16.dp),
        verticalArrangement = Arrangement.spacedBy(16.dp)
    ) {
        Text("LAN SYNC", style = RetroTypography.headlineMedium)

        // Connection status
        RetroCard {
            Row(verticalAlignment = Alignment.CenterVertically) {
                val (statusText, statusColor) = when (connectionState) {
                    ConnectionState.CONNECTED -> "● CONNECTED" to RetroGreen
                    ConnectionState.CONNECTING -> "◌ CONNECTING..." to RetroAmber
                    ConnectionState.DISCONNECTED -> "○ OFFLINE" to RetroRed
                }
                Text(statusText, style = RetroTypography.titleMedium, color = statusColor)
            }
        }

        // Player 2 join animation header
        if (connectionState == ConnectionState.CONNECTED) {
            RetroCard {
                Text(
                    "★ PLAYER 2 HAS JOINED ★",
                    style = RetroTypography.titleMedium,
                    color = RetroAmber
                )
            }
        }

        // Discovered peers
        Text("NEARBY DEVICES", style = RetroTypography.titleMedium)

        if (peers.isEmpty()) {
            RetroCard {
                Text(
                    "Scanning LAN...\nMake sure Homie is running on your desktop.",
                    style = RetroTypography.bodyMedium,
                    color = RetroCyan
                )
            }
        } else {
            LazyColumn(verticalArrangement = Arrangement.spacedBy(8.dp)) {
                items(peers) { peer ->
                    RetroCard(
                        modifier = Modifier.clickable { viewModel.connectToPeer(peer) }
                    ) {
                        Column {
                            Text(peer.name, style = RetroTypography.titleMedium, color = RetroGreen)
                            Text(
                                "${peer.host}:${peer.port}",
                                style = RetroTypography.bodyMedium,
                                color = RetroGray
                            )
                            Text("[CONNECT]", style = RetroTypography.labelMedium, color = RetroAmber)
                        }
                    }
                }
            }
        }

        // Disconnect button
        if (connectionState == ConnectionState.CONNECTED) {
            Text(
                "[DISCONNECT]",
                style = RetroTypography.labelMedium,
                color = RetroRed,
                modifier = Modifier
                    .clickable { viewModel.disconnect() }
                    .pixelBorder(RetroRed, width = 1f)
                    .padding(8.dp)
            )
        }

        ScanlineOverlay(alpha = 0.03f)
    }
}
```

- [ ] **Step 3: Build and verify**

Run: `cd android && ./gradlew assembleDebug`
Expected: BUILD SUCCESSFUL

- [ ] **Step 4: Commit**

```bash
git add android/app/src/main/kotlin/com/heyhomie/app/ui/
git commit -m "feat(android): add LAN network screen with peer discovery and pairing UI"
```

---

### Task 17: Settings Screen

**Files:**
- Modify: `android/app/src/main/kotlin/com/heyhomie/app/ui/screens/SettingsScreen.kt`
- Create: `android/app/src/main/kotlin/com/heyhomie/app/core/config/SettingsStore.kt`
- Create: `android/app/src/main/kotlin/com/heyhomie/app/ui/viewmodel/SettingsViewModel.kt`

- [ ] **Step 1: Create SettingsStore (DataStore)**

```kotlin
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
```

- [ ] **Step 2: Create SettingsViewModel**

```kotlin
package com.heyhomie.app.ui.viewmodel

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.heyhomie.app.core.config.SettingsStore
import dagger.hilt.android.lifecycle.HiltViewModel
import kotlinx.coroutines.flow.SharingStarted
import kotlinx.coroutines.flow.stateIn
import kotlinx.coroutines.launch
import javax.inject.Inject

@HiltViewModel
class SettingsViewModel @Inject constructor(
    private val settings: SettingsStore
) : ViewModel() {
    val scanlines = settings.scanlines.stateIn(viewModelScope, SharingStarted.Eagerly, true)
    val highContrast = settings.highContrast.stateIn(viewModelScope, SharingStarted.Eagerly, false)
    val soundEffects = settings.soundEffects.stateIn(viewModelScope, SharingStarted.Eagerly, true)
    val syncScope = settings.syncScope.stateIn(viewModelScope, SharingStarted.Eagerly, "all")

    fun toggleScanlines() = viewModelScope.launch { settings.setScanlines(!scanlines.value) }
    fun toggleHighContrast() = viewModelScope.launch { settings.setHighContrast(!highContrast.value) }
    fun toggleSoundEffects() = viewModelScope.launch { settings.setSoundEffects(!soundEffects.value) }
    fun setSyncScope(scope: String) = viewModelScope.launch { settings.setSyncScope(scope) }
}
```

- [ ] **Step 3: Replace SettingsScreen**

```kotlin
package com.heyhomie.app.ui.screens

import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Text
import androidx.compose.runtime.*
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import androidx.hilt.navigation.compose.hiltViewModel
import com.heyhomie.app.ui.components.*
import com.heyhomie.app.ui.theme.*
import com.heyhomie.app.ui.viewmodel.SettingsViewModel

@Composable
fun SettingsScreen(viewModel: SettingsViewModel = hiltViewModel()) {
    val scanlines by viewModel.scanlines.collectAsState()
    val highContrast by viewModel.highContrast.collectAsState()
    val soundEffects by viewModel.soundEffects.collectAsState()
    val syncScope by viewModel.syncScope.collectAsState()

    Column(
        modifier = Modifier
            .fillMaxSize()
            .background(RetroDark)
            .verticalScroll(rememberScrollState())
            .padding(16.dp),
        verticalArrangement = Arrangement.spacedBy(12.dp)
    ) {
        Text("CONFIG", style = RetroTypography.headlineMedium)

        // Visual
        Text("DISPLAY", style = RetroTypography.titleMedium)
        RetroToggle("Scanline overlay", scanlines) { viewModel.toggleScanlines() }
        RetroToggle("High contrast mode", highContrast) { viewModel.toggleHighContrast() }
        RetroToggle("8-bit sound FX", soundEffects) { viewModel.toggleSoundEffects() }

        Spacer(Modifier.height(8.dp))

        // Sync
        Text("SYNC SCOPE", style = RetroTypography.titleMedium)
        listOf("all" to "ALL MEMORY", "conversations" to "CONVERSATIONS ONLY", "manual" to "MANUAL")
            .forEach { (value, label) ->
                val selected = syncScope == value
                Text(
                    text = "${if (selected) "►" else " "} $label",
                    style = RetroTypography.bodyMedium,
                    color = if (selected) RetroGreen else RetroGray,
                    modifier = Modifier
                        .clickable { viewModel.setSyncScope(value) }
                        .padding(vertical = 4.dp)
                )
            }

        Spacer(Modifier.height(8.dp))

        // About
        Text("ABOUT", style = RetroTypography.titleMedium)
        RetroCard {
            Column {
                Text("HOMIE AI v0.1.0", style = RetroTypography.bodyMedium, color = RetroGreen)
                Text("Local-first AI assistant", style = RetroTypography.bodyMedium, color = RetroGray)
                Text("heyhomie.ai", style = RetroTypography.bodyMedium, color = RetroCyan)
            }
        }

        ScanlineOverlay(alpha = 0.03f)
    }
}

@Composable
private fun RetroToggle(label: String, enabled: Boolean, onToggle: () -> Unit) {
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .clickable(onClick = onToggle)
            .pixelBorder(if (enabled) RetroGreen else RetroGray, width = 1f)
            .padding(12.dp),
        horizontalArrangement = Arrangement.SpaceBetween
    ) {
        Text(label, style = RetroTypography.bodyMedium)
        Text(
            if (enabled) "[ON]" else "[OFF]",
            style = RetroTypography.labelMedium,
            color = if (enabled) RetroGreen else RetroRed
        )
    }
}
```

- [ ] **Step 4: Build and verify**

Run: `cd android && ./gradlew assembleDebug`
Expected: BUILD SUCCESSFUL

- [ ] **Step 5: Commit**

```bash
git add android/app/src/main/kotlin/com/heyhomie/app/
git commit -m "feat(android): add settings screen with DataStore and retro toggle controls"
```

---

### Task 18: Wire Inference Router to Chat

**Files:**
- Modify: `android/app/src/main/kotlin/com/heyhomie/app/ui/viewmodel/ChatViewModel.kt`
- Create: `android/app/src/main/kotlin/com/heyhomie/app/core/inference/di/InferenceModule.kt`

- [ ] **Step 1: Create InferenceModule**

```kotlin
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
```

- [ ] **Step 2: Update ChatViewModel to use InferenceRouter**

```kotlin
package com.heyhomie.app.ui.viewmodel

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.heyhomie.app.core.inference.InferenceRouter
import com.heyhomie.app.core.model.ChatMessage
import com.heyhomie.app.core.model.MessageRole
import dagger.hilt.android.lifecycle.HiltViewModel
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch
import javax.inject.Inject

@HiltViewModel
class ChatViewModel @Inject constructor(
    private val inferenceRouter: InferenceRouter
) : ViewModel() {
    private val _messages = MutableStateFlow<List<ChatMessage>>(emptyList())
    val messages: StateFlow<List<ChatMessage>> = _messages.asStateFlow()

    private val _inputText = MutableStateFlow("")
    val inputText: StateFlow<String> = _inputText.asStateFlow()

    private val _isGenerating = MutableStateFlow(false)
    val isGenerating: StateFlow<Boolean> = _isGenerating.asStateFlow()

    val fallbackBanner: String? get() = inferenceRouter.fallbackBanner
    val inferenceSource: String get() = inferenceRouter.activeSourceName

    fun updateInput(text: String) { _inputText.value = text }

    fun sendMessage(text: String) {
        if (text.isBlank() || _isGenerating.value) return
        val userMsg = ChatMessage(role = MessageRole.USER, text = text.trim())
        _messages.value = _messages.value + userMsg
        _inputText.value = ""

        viewModelScope.launch {
            _isGenerating.value = true
            try {
                val response = inferenceRouter.generate(
                    prompt = text.trim(),
                    systemPrompt = "You are Homie, a friendly local-first AI assistant. Be helpful, concise, and warm."
                )
                val assistantMsg = ChatMessage(
                    role = MessageRole.ASSISTANT,
                    text = response,
                    isStreaming = true
                )
                _messages.value = _messages.value + assistantMsg
            } catch (e: Exception) {
                val errorMsg = ChatMessage(
                    role = MessageRole.SYSTEM,
                    text = "ERROR: ${e.message ?: "Inference failed"}"
                )
                _messages.value = _messages.value + errorMsg
            } finally {
                _isGenerating.value = false
            }
        }
    }
}
```

- [ ] **Step 3: Build and verify**

Run: `cd android && ./gradlew assembleDebug`
Expected: BUILD SUCCESSFUL

- [ ] **Step 4: Commit**

```bash
git add android/app/src/main/kotlin/com/heyhomie/app/
git commit -m "feat(android): wire inference router to chat with Qubrid fallback"
```

---

### Task 19: Fallback Banner Display

**Files:**
- Modify: `android/app/src/main/kotlin/com/heyhomie/app/ui/screens/ChatScreen.kt`

- [ ] **Step 1: Add fallback banner to ChatScreen**

Add at the top of the Column, before LazyColumn:
```kotlin
val fallbackBanner = viewModel.fallbackBanner
if (fallbackBanner != null) {
    Text(
        text = fallbackBanner,
        style = RetroTypography.labelMedium,
        color = RetroAmber,
        modifier = Modifier
            .fillMaxWidth()
            .background(RetroDarkCard)
            .padding(8.dp)
    )
}
```

Also add inference source label:
```kotlin
Text(
    text = "SOURCE: ${viewModel.inferenceSource}",
    style = RetroTypography.labelMedium,
    color = RetroCyan,
    modifier = Modifier.padding(horizontal = 12.dp, vertical = 4.dp)
)
```

- [ ] **Step 2: Build and verify**

Run: `cd android && ./gradlew assembleDebug`
Expected: BUILD SUCCESSFUL

- [ ] **Step 3: Commit**

```bash
git add android/app/src/main/kotlin/com/heyhomie/app/ui/screens/ChatScreen.kt
git commit -m "feat(android): display inference source and fallback banner in chat"
```

---

### Task 20: Final Integration — Gradle Wrapper & .gitignore

**Files:**
- Create: `android/.gitignore`

- [ ] **Step 1: Create .gitignore**

```
*.iml
.gradle
/local.properties
/.idea
.DS_Store
/build
/captures
.externalNativeBuild
.cxx
local.properties
/app/build
```

- [ ] **Step 2: Generate Gradle wrapper**

Run: `cd android && gradle wrapper --gradle-version 8.11`

- [ ] **Step 3: Full build verification**

Run: `cd android && ./gradlew clean assembleDebug`
Expected: BUILD SUCCESSFUL

- [ ] **Step 4: Run all unit tests**

Run: `cd android && ./gradlew testDebugUnitTest`
Expected: All tests pass

- [ ] **Step 5: Commit**

```bash
git add android/
git commit -m "feat(android): finalize project setup with wrapper and gitignore"
```
