package com.heyhomie.app.ui.components

import androidx.compose.foundation.layout.*
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
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
