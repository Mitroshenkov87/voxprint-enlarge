# vendored, to be replaced by the voxprint-common dependency
"""Shared 960 by 540 splash.

Artwork covers the window, clipped to a 12 px corner radius. When the screen
device pixel ratio is at least 2 and ``<stem>@2x<suffix>`` exists beside the
1x file, that file is used. The left third is inset 48 px and holds the app
name, the version line, the stage line, and a 4 px bar along the bottom. The
bar shimmers until ``set_progress`` is given a fraction. ``finish`` holds the
splash for at least 600 ms after ``show``, then fades it out over 200 ms.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

from PySide6.QtCore import QEasingCurve, QEventLoop, QPoint, QPropertyAnimation, QRect, QRectF, Qt, QTimer
from PySide6.QtGui import QColor, QFont, QGuiApplication, QPainter, QPainterPath, QPixmap
from PySide6.QtWidgets import QApplication, QGraphicsDropShadowEffect, QWidget

WIDTH = 960
HEIGHT = 540
PADDING = 48
RADIUS = 12
BAR_HEIGHT = 4
MIN_VISIBLE_S = 0.6
FADE_MS = 200
_OFF_WHITE = QColor("#f2f2f5")
_FONT = ("Segoe UI Variable Text", "Segoe UI", "Ubuntu", "Noto Sans", "Cantarell", "DejaVu Sans")


def choose_art(path: Path, ratio: float) -> Path:
    """The 2x file when ``ratio`` is at least 2 and that file exists, otherwise ``path``."""
    if ratio >= 2:
        hidpi = path.with_name(f"{path.stem}@2x{path.suffix}")
        if hidpi.is_file():
            return hidpi
    return path


def format_splash_version(payload: dict[str, Any], build: int) -> str:
    """``0.2.0 RC 5 build 12 "Chayil"``, or the same line without ``RC N`` for a final version."""
    version = str(payload.get("version") or "").strip()
    head, rc = _split_version(version)
    codename = str(payload.get("codename") or "").strip()
    shown = f"{head} RC {rc}" if head and rc else (head or version)
    line = f"{shown} build {build}".strip()
    if codename:
        line += f' "{codename}"'
    return line


def resolve_build(payload: dict[str, Any], directory: Path | None, env: dict[str, str] | None = None) -> int:
    """Build number from ``BUILD.json``, then a sibling ``build_info.json``, then the environment."""
    raw = payload.get("build")
    if isinstance(raw, int) and not isinstance(raw, bool):
        return raw
    if directory is not None:
        stamped = _read_json(directory / "build_info.json")
        stamped_build = stamped.get("build")
        if isinstance(stamped_build, int) and not isinstance(stamped_build, bool):
            return stamped_build
    source = os.environ if env is None else env
    override = str(source.get("VOXPRINT_BUILD", "")).strip()
    if override.isdigit():
        return int(override)
    run = str(source.get("GITHUB_RUN_NUMBER", "")).strip()
    offset = payload.get("offset", 0)
    try:
        extra = int(offset) if not isinstance(offset, bool) else 0
    except (TypeError, ValueError):
        extra = 0
    if run.isdigit():
        return int(run) + extra
    return 0


def _split_version(version: str) -> tuple[str, str]:
    base, marker, rc = version.partition("-rc.")
    if marker and base[:1].isdigit() and rc.isdigit() and base.count(".") == 2:
        return base, rc
    return version, ""


def _read_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def _device_ratio() -> float:
    screen = QGuiApplication.primaryScreen()
    if screen is None:
        return 1.0
    return float(screen.devicePixelRatio())


class SplashScreen(QWidget):
    """Borderless centred splash. Create and ``show`` it before the heavy imports."""

    def __init__(
        self,
        app_id: str,
        art_path: Path,
        identity_color: str,
        build_json_path: Path | None = None,
    ) -> None:
        """Load ``BUILD.json`` and the artwork. The window stays hidden until ``show``.

        Args:
            app_id: Text painted as the app name.
            art_path: 1x artwork. A sibling ``<stem>@2x<suffix>`` is used when
                the screen device pixel ratio is at least 2 and that file exists.
            identity_color: Fill colour of the progress bar.
            build_json_path: ``BUILD.json``. The version line is read from this
                file and a sibling ``build_info.json``.
        """
        super().__init__(None)
        self.app_id = app_id
        self._art_path = Path(art_path)
        self._identity = QColor(identity_color)
        self._build_path = Path(build_json_path) if build_json_path is not None else None
        payload = _read_json(self._build_path) if self._build_path is not None else {}
        directory = self._build_path.parent if self._build_path is not None else None
        self.version_text = format_splash_version(payload, resolve_build(payload, directory))
        self._stage = ""
        self._fraction: float | None = None
        self._phase = 0.0
        self._shown_at: float | None = None
        self._finished = False
        self._art_used = self._art_path
        self._pixmap = QPixmap()
        self._cover: QPixmap | None = None
        self._cover_ratio = 0.0
        self.setFixedSize(WIDTH, HEIGHT)
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint | Qt.WindowType.Tool
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        shadow = QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(28)
        shadow.setOffset(0, 10)
        shadow.setColor(QColor(0, 0, 0, 140))
        self.setGraphicsEffect(shadow)
        self._timer = QTimer(self)
        self._timer.setInterval(30)
        self._timer.timeout.connect(self._tick)

    def show(self) -> None:
        """Show the splash, centred, and start the indeterminate bar."""
        self._art_used = choose_art(self._art_path, _device_ratio())
        self._pixmap = QPixmap(str(self._art_used))
        self._cover = None
        self._place()
        if self._fraction is None:
            self._timer.start()
        super().show()
        self.raise_()
        if self._shown_at is None:
            self._shown_at = time.monotonic()
        app = QApplication.instance()
        if app is not None:
            app.processEvents()

    def set_stage(self, text: str) -> None:
        """Replace the stage line and repaint."""
        self._stage = text
        self.update()
        app = QApplication.instance()
        if app is not None:
            app.processEvents()

    def set_progress(self, fraction: float | None) -> None:
        """Shimmer when ``fraction`` is ``None``. A number fills that much of the bar."""
        if fraction is None:
            self._fraction = None
            if self.isVisible() and not self._timer.isActive():
                self._timer.start()
        else:
            self._fraction = max(0.0, min(1.0, float(fraction)))
            self._timer.stop()
        self.update()
        app = QApplication.instance()
        if app is not None:
            app.processEvents()

    def finish(self, main_window: QWidget) -> None:
        """Hold until 600 ms from ``show``, fade out over 200 ms, then leave ``main_window`` up."""
        if self._finished:
            return
        self._finished = True
        if self._shown_at is None:
            self._shown_at = time.monotonic()
        remain = MIN_VISIBLE_S - (time.monotonic() - self._shown_at)
        if remain > 0:
            self._wait_ms(int(remain * 1000))
        self._timer.stop()
        main_window.show()
        self._fade()
        self.close()

    def closeEvent(self, event) -> None:  # noqa: N802, ANN001
        """Stop the shimmer timer when the window closes."""
        self._timer.stop()
        super().closeEvent(event)

    def __enter__(self) -> SplashScreen:
        """Show the splash."""
        self.show()
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        """Close the splash when ``finish`` has not already run."""
        if not self._finished:
            self._timer.stop()
            self.close()
            self._finished = True

    def paintEvent(self, event) -> None:  # noqa: N802, ANN001
        """Paint the cover, the left-third text, and the bar."""
        del event
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        clip = QPainterPath()
        clip.addRoundedRect(QRectF(self.rect()), RADIUS, RADIUS)
        painter.setClipPath(clip)
        self._paint_art(painter)
        self._paint_text(painter)
        self._paint_bar(painter)
        painter.end()

    def _place(self) -> None:
        screen = QGuiApplication.primaryScreen()
        area = screen.availableGeometry() if screen is not None else QRect(0, 0, WIDTH, HEIGHT)
        self.move(area.center() - QPoint(WIDTH // 2, HEIGHT // 2))

    def _cover_pixmap(self) -> QPixmap:
        ratio = _device_ratio()
        if self._cover is not None and self._cover_ratio == ratio:
            return self._cover
        scaled = self._pixmap.scaled(
            int(WIDTH * ratio),
            int(HEIGHT * ratio),
            Qt.AspectRatioMode.KeepAspectRatioByExpanding,
            Qt.TransformationMode.SmoothTransformation,
        )
        scaled.setDevicePixelRatio(ratio)
        self._cover = scaled
        self._cover_ratio = ratio
        return scaled

    def _paint_art(self, painter: QPainter) -> None:
        if self._pixmap.isNull():
            painter.fillRect(self.rect(), QColor("#17171c"))
            return
        scaled = self._cover_pixmap()
        ratio = scaled.devicePixelRatio() or 1.0
        x = (WIDTH - scaled.width() / ratio) / 2
        y = (HEIGHT - scaled.height() / ratio) / 2
        painter.drawPixmap(QPoint(int(x), int(y)), scaled)

    def _paint_text(self, painter: QPainter) -> None:
        column = WIDTH // 3 - PADDING
        name = QFont()
        name.setFamilies(list(_FONT))
        name.setPixelSize(40)
        name.setWeight(QFont.Weight.DemiBold)
        painter.setFont(name)
        painter.setPen(_OFF_WHITE)
        flags = Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop | Qt.TextFlag.TextWordWrap
        name_rect = painter.fontMetrics().boundingRect(QRect(PADDING, PADDING, column, 160), flags, self.app_id)
        painter.drawText(name_rect, flags, self.app_id)
        version = QFont()
        version.setFamilies(list(_FONT))
        version.setPixelSize(16)
        painter.setFont(version)
        muted = QColor(_OFF_WHITE)
        muted.setAlpha(178)
        painter.setPen(muted)
        version_rect = QRect(PADDING, name_rect.bottom() + 10, column, 24)
        painter.drawText(version_rect, Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, self.version_text)
        stage = QFont()
        stage.setFamilies(list(_FONT))
        stage.setPixelSize(14)
        painter.setFont(stage)
        painter.setPen(_OFF_WHITE)
        stage_flags = Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop | Qt.TextFlag.TextWordWrap
        stage_rect = painter.fontMetrics().boundingRect(
            QRect(PADDING, version_rect.bottom() + 8, column, 80),
            stage_flags,
            self._stage,
        )
        painter.drawText(stage_rect, stage_flags, self._stage)

    def _paint_bar(self, painter: QPainter) -> None:
        bar = QRectF(PADDING, HEIGHT - PADDING - BAR_HEIGHT, WIDTH // 3 - PADDING, BAR_HEIGHT)
        track = QColor(_OFF_WHITE)
        track.setAlpha(51)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(track)
        painter.drawRoundedRect(bar, 2, 2)
        if self._fraction is None:
            span = bar.width() * 0.35
            travel = bar.width() + span
            x = bar.left() - span + self._phase * travel
            shine = QColor(_OFF_WHITE)
            shine.setAlpha(110)
            painter.setBrush(shine)
            painter.save()
            painter.setClipRect(bar)
            painter.drawRoundedRect(QRectF(x, bar.top(), span, bar.height()), 2, 2)
            painter.restore()
            return
        fill = bar.width() * self._fraction
        if fill <= 0:
            return
        painter.setBrush(self._identity)
        painter.drawRoundedRect(QRectF(bar.left(), bar.top(), fill, bar.height()), 2, 2)

    def _tick(self) -> None:
        self._phase = (self._phase + 0.03) % 1.0
        self.update()

    def _wait_ms(self, milliseconds: int) -> None:
        loop = QEventLoop()
        QTimer.singleShot(max(0, milliseconds), loop.quit)
        loop.exec()

    def _fade(self) -> None:
        self.setWindowOpacity(1.0)
        animation = QPropertyAnimation(self, b"windowOpacity", self)
        animation.setDuration(FADE_MS)
        animation.setStartValue(1.0)
        animation.setEndValue(0.0)
        animation.setEasingCurve(QEasingCurve.Type.Linear)
        loop = QEventLoop()
        animation.finished.connect(loop.quit)
        animation.start()
        self._animation = animation
        loop.exec()
