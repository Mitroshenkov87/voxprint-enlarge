"""Before/after compare. Drag the divider, scroll or use the buttons to zoom.

Both pictures are fitted to the same rectangle, so a 4x file lines up with the original.
"""
from __future__ import annotations

from pathlib import Path

from PIL import Image
from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QFont, QImage, QMouseEvent, QPainter, QPen, QPixmap, QWheelEvent
from PySide6.QtWidgets import QWidget

from voxprint_engine.formats import open_rgb

_MAX_EDGE = 4096


def _load(path: Path | None) -> QPixmap:
    """Decode with Pillow so AVIF and JPEG XL show up. Qt cannot read those files."""
    if path is None or not path.is_file():
        return QPixmap()
    try:
        frame = open_rgb(path)
    except (OSError, ValueError):
        return QPixmap()
    if max(frame.size) > _MAX_EDGE:
        frame.thumbnail((_MAX_EDGE, _MAX_EDGE), Image.Resampling.LANCZOS)
    rgb = frame.convert("RGB")
    data = rgb.tobytes("raw", "RGB")
    qimage = QImage(data, rgb.width, rgb.height, rgb.width * 3, QImage.Format.Format_RGB888)
    return QPixmap.fromImage(qimage.copy())


class CompareView(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.setMinimumHeight(260)
        self.setMouseTracking(True)
        self.before = QPixmap()
        self.after = QPixmap()
        self.split = 0.55
        self.zoom = 1.0
        self.pan = QPointF(0, 0)
        self._mode = ""
        self._origin = QPointF()
        self._pan_origin = QPointF()

    def set_pair(self, before: Path | None, after: Path | None) -> None:
        self.before = _load(before)
        self.after = _load(after)
        self.zoom = 1.0
        self.pan = QPointF(0, 0)
        self.split = 0.55
        self.update()

    def zoom_by(self, factor: float) -> None:
        self.zoom = min(8.0, max(1.0, self.zoom * factor))
        if self.zoom <= 1.0:
            self.zoom = 1.0
            self.pan = QPointF(0, 0)
        self.update()

    def zoom_fit(self) -> None:
        self.zoom = 1.0
        self.pan = QPointF(0, 0)
        self.update()

    def zoom_text(self) -> str:
        return f"{int(round(self.zoom * 100))}%"

    def _source(self) -> QPixmap:
        return self.before if not self.before.isNull() else self.after

    def _dest(self) -> QRectF:
        source = self._source()
        if source.isNull():
            return QRectF()
        avail = QRectF(self.rect().adjusted(8, 8, -8, -8))
        fit = min(avail.width() / source.width(), avail.height() / source.height())
        width = source.width() * fit * self.zoom
        height = source.height() * fit * self.zoom
        x = avail.center().x() - width / 2 + self.pan.x()
        y = avail.center().y() - height / 2 + self.pan.y()
        return QRectF(x, y, width, height)

    def _cut(self, dest: QRectF) -> float:
        return dest.left() + dest.width() * self.split

    def paintEvent(self, _event: object) -> None:  # noqa: N802 - Qt name
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        painter.fillRect(self.rect(), QColor("#14141c"))
        dest = self._dest()
        if dest.isNull():
            painter.setPen(QColor("#a8a8b8"))
            font = QFont(self.font())
            font.setPixelSize(14)
            painter.setFont(font)
            painter.drawText(self.rect().adjusted(24, 24, -24, -24), Qt.AlignmentFlag.AlignCenter,
                             "Run an upscale to compare before and after.\nDrag the divider. Scroll to zoom.")
            painter.end()
            return
        if not self.before.isNull():
            painter.drawPixmap(dest, self.before, QRectF(self.before.rect()))
        cut = self._cut(dest)
        if not self.after.isNull():
            painter.save()
            painter.setClipRect(QRectF(cut, dest.top(), max(0.0, dest.right() - cut), dest.height()))
            painter.drawPixmap(dest, self.after, QRectF(self.after.rect()))
            painter.restore()
        painter.setPen(QPen(QColor("#f2f2f5"), 2))
        painter.drawLine(QPointF(cut, dest.top()), QPointF(cut, dest.bottom()))
        painter.setBrush(QColor("#60a5fa"))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawRoundedRect(QRectF(cut - 5, dest.center().y() - 16, 10, 32), 3, 3)
        painter.setPen(QColor("#0b1220"))
        font = QFont(self.font())
        font.setPixelSize(11)
        font.setBold(True)
        painter.setFont(font)
        if not self.before.isNull():
            painter.fillRect(QRectF(dest.left() + 8, dest.top() + 8, 62, 20), QColor(10, 12, 18, 180))
            painter.setPen(QColor("#f2f2f5"))
            painter.drawText(QRectF(dest.left() + 8, dest.top() + 8, 62, 20), Qt.AlignmentFlag.AlignCenter, "Before")
        if not self.after.isNull():
            painter.fillRect(QRectF(dest.right() - 70, dest.top() + 8, 62, 20), QColor(10, 12, 18, 180))
            painter.setPen(QColor("#f2f2f5"))
            painter.drawText(QRectF(dest.right() - 70, dest.top() + 8, 62, 20), Qt.AlignmentFlag.AlignCenter, "After")
        painter.end()

    def wheelEvent(self, event: QWheelEvent) -> None:  # noqa: N802 - Qt name
        self.zoom_by(1.12 if event.angleDelta().y() > 0 else 1 / 1.12)
        event.accept()

    def mousePressEvent(self, event: QMouseEvent) -> None:  # noqa: N802 - Qt name
        dest = self._dest()
        if dest.isNull() or event.button() != Qt.MouseButton.LeftButton:
            return
        near = abs(event.position().x() - self._cut(dest)) < 14
        self._mode = "split" if near else "pan"
        self._origin = event.position()
        self._pan_origin = QPointF(self.pan)
        event.accept()

    def mouseMoveEvent(self, event: QMouseEvent) -> None:  # noqa: N802 - Qt name
        dest = self._dest()
        if self._mode == "split" and not dest.isNull() and dest.width() > 1:
            rel = (event.position().x() - dest.left()) / dest.width()
            self.split = min(0.95, max(0.05, rel))
            self.update()
        elif self._mode == "pan":
            delta = event.position() - self._origin
            self.pan = self._pan_origin + delta
            self.update()
        elif not dest.isNull():
            near = abs(event.position().x() - self._cut(dest)) < 14
            self.setCursor(Qt.CursorShape.SplitHCursor if near else Qt.CursorShape.ArrowCursor)
        event.accept()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:  # noqa: N802 - Qt name
        self._mode = ""
        event.accept()
