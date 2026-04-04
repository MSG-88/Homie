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
    val backStackEntry by navController.currentBackStackEntryAsState()
    val currentRoute = backStackEntry?.destination?.route
    Scaffold(bottomBar = {
        if (currentRoute != null && currentRoute != "boot") {
            GameHudNavBar(currentRoute = currentRoute, onNavigate = { screen ->
                navController.navigate(screen.route) { popUpTo(Screen.Chat.route) { saveState = true }; launchSingleTop = true; restoreState = true }
            })
        }
    }) { innerPadding ->
        NavHost(navController = navController, startDestination = "boot", modifier = Modifier.padding(innerPadding)) {
            composable("boot") { BootScreen(onBootComplete = { navController.navigate(Screen.Chat.route) { popUpTo("boot") { inclusive = true } } }) }
            composable(Screen.Chat.route) { ChatScreen() }
            composable(Screen.PhoneStats.route) { PhoneStatsScreen() }
            composable(Screen.Network.route) { NetworkScreen() }
            composable(Screen.Settings.route) { SettingsScreen() }
        }
    }
}
