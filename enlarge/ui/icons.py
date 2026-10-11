"""Suite icons. They are 22px stroke SVGs that use currentColor."""
from __future__ import annotations

from PySide6.QtCore import QByteArray, Qt, QSize
from PySide6.QtGui import QIcon, QImage, QPainter, QPixmap
from PySide6.QtSvg import QSvgRenderer
from PySide6.QtWidgets import QPushButton

from enlarge.appinfo import resource_dir
from voxprint_common.ui.theme import TEXT, TEXT_ON_PRIMARY

_NAMES = ("save", "save-as", "save-copy", "export", "download")


def suite_icon(name: str, *, on_primary: bool = False) -> QIcon:
    """Rasterise one suite SVG in the theme text colour."""
    if name not in _NAMES:
        return QIcon()
    path = resource_dir() / "assets" / "icons" / f"{name}.svg"
    if not path.is_file():
        return QIcon()
    colour = TEXT_ON_PRIMARY if on_primary else TEXT
    svg = path.read_text(encoding="utf-8").replace("currentColor", colour)
    renderer = QSvgRenderer(QByteArray(svg.encode("utf-8")))
    image = QImage(22, 22, QImage.Format.Format_ARGB32_Premultiplied)
    image.fill(Qt.GlobalColor.transparent)
    painter = QPainter(image)
    renderer.render(painter)
    painter.end()
    return QIcon(QPixmap.fromImage(image))


def apply_suite_icon(button: QPushButton, name: str, *, on_primary: bool = False) -> None:
    """Put a suite icon on a button. The text label stays."""
    icon = suite_icon(name, on_primary=on_primary)
    if icon.isNull():
        return
    button.setIcon(icon)
    button.setIconSize(QSize(18, 18))
