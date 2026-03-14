package com.heyhomie.app.ui.components

import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.drawBehind
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.Paint
import androidx.compose.ui.graphics.drawscope.drawIntoCanvas
import androidx.compose.ui.graphics.nativeCanvas
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
