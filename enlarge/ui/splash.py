"""Start-up splash. Same layout as the other Voxprint apps: square art, a strip with the name and a thin bar.

The hidden splash picture used by Movie Dubber is not bundled here. It can be added later without changing this layout.
"""
from __future__ import annotations

from PySide6.QtCore import QRect, QRectF, Qt
from PySide6.QtGui import QColor, QFont, QGuiApplication, QPainter, QPixmap
from PySide6.QtWidgets import QApplication, QSplashScreen

from enlarge.appinfo import APP_DISPLAY_NAME, resource_dir, version_label

SIDE = 480
STRIP = 92
TEXT, MUTED, BAR = "#f2f2f5", "#d0d0da", "#ff6b8a"


def _side() -> int:
    screen = QGuiApplication.primaryScreen()
    if screen is None:
        return SIDE
    area = screen.availableGeometry()
    return max(240, min(SIDE, int(min(area.width(), area.height()) * 0.6)))


def _pixmap(side: int) -> QPixmap:
    screen = QGuiApplication.primaryScreen()
    ratio = screen.devicePixelRatio() if screen is not None else 1.0
    art = QPixmap(str(resource_dir() / "assets" / "splash.jpg"))
    if art.isNull():
        art = QPixmap(int(side * ratio), int(side * ratio))
        art.fill(QColor("#070b18"))
    image = art.scaled(
        int(side * ratio),
        int(side * ratio),
        Qt.AspectRatioMode.KeepAspectRatioByExpanding,
        Qt.TransformationMode.SmoothTransformation,
    )
    image.setDevicePixelRatio(ratio)
    return image


class Splash(QSplashScreen):
    """Startup splash. The bar colour stays pink."""
    def __init__(self) -> None:
        """Load the artwork and the version line."""
        self.side = _side()
        super().__init__(_pixmap(self.side))
        self._text = ""
        self._fraction = 0.0

    def drawContents(self, painter: QPainter) -> None:  # noqa: N802 - Qt name
        """Paint the mark, the title, and the progress bar."""
        side = self.side
        top = side - STRIP
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(QRect(0, top, side, STRIP), QColor(10, 12, 18, 190))
        font = QFont(self.font())
        font.setPixelSize(18)
        font.setBold(True)
        painter.setFont(font)
        painter.setPen(QColor(TEXT))
        painter.drawText(
            QRect(18, top + 10, side - 36, 26),
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
            APP_DISPLAY_NAME,
        )
        font.setPixelSize(12)
        font.setBold(False)
        painter.setFont(font)
        painter.setPen(QColor(MUTED))
        painter.drawText(
            QRect(18, top + 54, side - 36, 26),
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
            version_label(),
        )
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(255, 255, 255, 50))
        painter.drawRoundedRect(QRectF(18, top + 44, side - 36, 4), 2, 2)
        painter.setBrush(QColor(BAR))
        painter.drawRoundedRect(QRectF(18, top + 44, (side - 36) * self._fraction, 4), 2, 2)
        painter.setPen(QColor(MUTED))
        painter.drawText(
            QRect(18, top + 54, side - 36, 26),
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
            self._text,
        )

    def status(self, text: str, fraction: float = -1.0) -> None:
        """Set the status line and the bar fraction from 0 to 1."""
        self._text = text
        if fraction >= 0:
            self._fraction = max(0.0, min(1.0, fraction))
        self.repaint()
        QApplication.processEvents()


def show() -> Splash:
    """Create the splash, show it, and return it."""
    splash = Splash()
    splash.show()
    QApplication.processEvents()
    return splash
