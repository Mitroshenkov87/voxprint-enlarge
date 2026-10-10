"""Dark glass theme shared across the Voxprint suite.

Yellow primary actions, soft blue menus, lists and accents. The splash bar stays
Enlarge's own colour (``#ff6b8a`` in ``enlarge.ui.splash``).

Adapted from Voxprint AI Audiobook Builder (Apache-2.0, Copyright 2026
Aleksandr Mitroshenkov). Every text colour is opaque and passes WCAG AA
(>= 4.5:1) on its panel. Panels and controls are almost solid. Only an Acrylic
window background is translucent.
"""
from __future__ import annotations

TEXT = "#f2f2f5"
TEXT_MUTED = "#c4c4d0"
TEXT_FAINT = "#a8a8b8"
TEXT_DISABLED = "#8e8e9e"
TEXT_ON_ACCENT = "#0b1220"
ACCENT = "#60a5fa"
ACCENT_HOVER = "#7db9ff"
ACCENT_SOFT = "#93c5fd"
ACCENT_STRONG = "#2563eb"
ROOT_PLAIN = "#17171c"
ROOT_GLASS = "rgba(16,16,22,222)"
CARD_PLAIN = "#202029"
CARD_GLASS = "rgba(34,34,44,238)"
CONTROL_BG = "#2c2c38"
CONTROL_HOVER = "#383846"
CONTROL_BORDER = "#4b4b5c"
AMBER = "#fcd34d"
NOTE_BG = "#2a2418"
NOTE_BORDER = "#6b5720"
GREEN = "#86efac"
BADGE_GREEN_TEXT = "#0b1f12"
BADGE_AMBER_TEXT = "#251a02"
RED = "#fca5a5"
PRIMARY = "#f5d76e"
PRIMARY_HOVER = "#f9e291"
PRIMARY_BORDER = "#fbe8a6"
TEXT_ON_PRIMARY = "#1c1604"
PRIMARY_DISABLED_BG = "#36321f"
PRIMARY_DISABLED_TEXT = "#b9b08a"
MENU_BG = "#1c2535"
MENU_SELECTION = "#2f4c78"
# Suite "more transparency" tint. Enlarge keeps the default Acrylic tint.
ROOT_GLASS_MORE = "rgba(16,16,22,167)"
CARD_GLASS_MORE = "rgba(34,34,44,214)"

STYLE_TEMPLATE = """
* {{{{ font-family: "Segoe UI Variable Text", "Segoe UI", "Ubuntu", "Noto Sans", "Cantarell", "DejaVu Sans", sans-serif; font-size: 14px; color: {text}; }}}}
QWidget#root {{{{ background: {{root_bg}}; }}}}
QLabel#title {{{{ font-size: 26px; font-weight: 600; }}}}
QLabel#subtitle {{{{ color: {muted}; }}}}
QFrame#card {{{{ background: {{card_bg}}; border: 1px solid {border}; border-radius: 12px; }}}}
QFrame#card[primary="true"] {{{{ border: 2px solid {primary}; }}}}
QLabel#sectiontitle {{{{ font-size: 16px; font-weight: 600; background: transparent; }}}}
QLabel#fileLabel, QLabel#status {{{{ color: {muted}; background: transparent; }}}}
QLabel#hint {{{{ color: {faint}; font-size: 12px; background: transparent; }}}}
QLabel#warn {{{{ color: {amber}; font-size: 12px; background: transparent; }}}}
QLabel#ok {{{{ color: {green}; background: transparent; }}}}
QLabel#footer {{{{ color: {faint}; font-size: 12px; }}}}
QLabel#chip {{{{ color: {faint}; padding: 3px 8px; border-radius: 10px; font-size: 12px; background: #2a2a35; }}}}
QLabel#chip[state="ready"] {{{{ color: {on_accent}; background: {soft}; font-weight: 600; }}}}
QLabel#chip[state="soon"] {{{{ color: {faint}; background: #2a2a35; }}}}
QPushButton {{{{ background: {control}; border: 1px solid {border}; border-radius: 8px; padding: 8px 16px; }}}}
QPushButton:hover {{{{ background: {hover}; }}}}
QPushButton:pressed {{{{ background: #23232d; }}}}
QPushButton:disabled {{{{ color: {disabled}; background: #1f1f28; border-color: #34343f; }}}}
QPushButton#primary {{{{ background: {primary}; border: 1px solid {primary_border}; font-size: 17px; font-weight: 600; padding: 14px 20px; color: {on_primary}; }}}}
QPushButton#primary:hover {{{{ background: {primary_hover}; }}}}
QPushButton#primary:disabled {{{{ background: {primary_dis_bg}; color: {primary_dis_text}; border-color: #4a4430; }}}}
QPushButton#step {{{{ padding: 10px 12px; font-weight: 600; }}}}
QPushButton#step:checked {{{{ background: {primary}; border-color: {primary_border}; color: {on_primary}; }}}}
QPushButton#step:checked:hover {{{{ background: {primary_hover}; }}}}
QPushButton#step[attention="true"] {{{{ color: {amber}; border-color: {amber}; }}}}
QPushButton#step[attention="true"]:checked {{{{ color: {on_primary}; border-color: {amber}; }}}}
QPushButton#primary[big="true"] {{{{ font-size: 20px; padding: 16px 24px; }}}}
QPushButton#disclosure {{{{ background: transparent; border: none; padding: 4px 2px; font-weight: 600; font-size: 15px; }}}}
QPushButton#disclosure:hover {{{{ color: {soft}; background: transparent; }}}}
QTableWidget {{{{ background: #1b1b23; border: 1px solid {border}; border-radius: 8px; gridline-color: #34343f; selection-background-color: {strong}; }}}}
QHeaderView::section {{{{ background: {control}; color: {text}; border: none; border-right: 1px solid {border}; padding: 6px; }}}}
QSlider::groove:horizontal {{{{ height: 6px; background: {control}; border-radius: 3px; }}}}
QSlider::handle:horizontal {{{{ background: {accent}; width: 16px; margin: -6px 0; border-radius: 8px; }}}}
QDialog {{{{ background: {{root_bg}}; }}}}
QPushButton#gear {{{{ padding: 6px 12px; font-size: 18px; }}}}
QProgressBar {{{{ background: {control}; border: 1px solid {border}; border-radius: 8px; height: 16px; text-align: center; }}}}
QProgressBar::chunk {{{{ background: {strong}; border-radius: 7px; }}}}
QComboBox {{{{ background: {control}; border: 1px solid {border}; border-radius: 8px; padding: 6px 12px; min-width: 130px; }}}}
QComboBox QAbstractItemView {{{{ background: {menu_bg}; border: 1px solid {border}; selection-background-color: {menu_sel}; selection-color: #ffffff; }}}}
QLineEdit {{{{ background: {control}; border: 1px solid {border}; border-radius: 8px; padding: 6px 12px; selection-background-color: {strong}; }}}}
QPlainTextEdit {{{{ background: #1b1b23; border: 1px solid {border}; border-radius: 8px; padding: 6px; font-family: "Cascadia Mono", "Consolas", "DejaVu Sans Mono", monospace; font-size: 12px; }}}}
QCheckBox {{{{ spacing: 8px; background: transparent; }}}}
QCheckBox::indicator {{{{ width: 16px; height: 16px; border: 1px solid {accent}; border-radius: 4px; background: {control}; }}}}
QCheckBox::indicator:checked {{{{ background: {strong}; border-color: {soft}; }}}}
QScrollArea {{{{ background: transparent; border: none; }}}}
QWidget#content {{{{ background: transparent; }}}}
QScrollBar:vertical {{{{ background: transparent; width: 10px; margin: 2px; }}}}
QScrollBar::handle:vertical {{{{ background: #5a5a6c; border-radius: 4px; min-height: 30px; }}}}
QScrollBar::add-line, QScrollBar::sub-line {{{{ width: 0; height: 0; }}}}
QMenu {{{{ background: {menu_bg}; border: 1px solid {border}; padding: 4px; }}}}
QMenu::item {{{{ padding: 5px 18px; border-radius: 6px; }}}}
QMenu::item:selected {{{{ background: {menu_sel}; color: #ffffff; }}}}
QListWidget {{{{ background: {menu_bg}; border: 1px solid {border}; border-radius: 8px; padding: 4px; }}}}
QListWidget::item {{{{ padding: 8px 10px; border-radius: 6px; }}}}
QListWidget::item:selected {{{{ background: {menu_sel}; color: #ffffff; }}}}
""".format(text=TEXT, muted=TEXT_MUTED, faint=TEXT_FAINT, disabled=TEXT_DISABLED, on_accent=TEXT_ON_ACCENT, accent=ACCENT,
           soft=ACCENT_SOFT, strong=ACCENT_STRONG, control=CONTROL_BG, hover=CONTROL_HOVER,
           border=CONTROL_BORDER, amber=AMBER, green=GREEN, primary=PRIMARY, primary_hover=PRIMARY_HOVER,
           primary_border=PRIMARY_BORDER, on_primary=TEXT_ON_PRIMARY, primary_dis_bg=PRIMARY_DISABLED_BG,
           primary_dis_text=PRIMARY_DISABLED_TEXT, menu_bg=MENU_BG, menu_sel=MENU_SELECTION)

#: (foreground, background) pairs that must keep >= 4.5:1. Checked by tests/test_app.py.
CONTRAST_PAIRS = [
    (TEXT, ROOT_PLAIN), (TEXT, CARD_PLAIN), (TEXT, CONTROL_BG), (TEXT, CONTROL_HOVER),
    (TEXT_MUTED, ROOT_PLAIN), (TEXT_MUTED, CARD_PLAIN), (TEXT_FAINT, ROOT_PLAIN), (TEXT_FAINT, CARD_PLAIN),
    (TEXT_ON_ACCENT, ACCENT), (TEXT_ON_ACCENT, ACCENT_HOVER), (TEXT_ON_ACCENT, ACCENT_SOFT),
    (TEXT, ACCENT_STRONG), ("#dcdce4", "#34343f"), (GREEN, CARD_PLAIN),
    (BADGE_GREEN_TEXT, GREEN), (BADGE_AMBER_TEXT, AMBER), (AMBER, CARD_PLAIN), (AMBER, NOTE_BG), (AMBER, ROOT_PLAIN),
    (RED, CARD_PLAIN), (TEXT_FAINT, "#2a2a35"), ("#ffffff", ACCENT_STRONG), (TEXT, "#4a3a12"),
    (AMBER, CONTROL_BG), (AMBER, CONTROL_HOVER), (ACCENT_SOFT, ROOT_PLAIN), (ACCENT_SOFT, CARD_PLAIN),
    (TEXT_ON_PRIMARY, PRIMARY), (TEXT_ON_PRIMARY, PRIMARY_HOVER), (PRIMARY_DISABLED_TEXT, PRIMARY_DISABLED_BG),
    (TEXT, MENU_BG), ("#ffffff", MENU_SELECTION),
]


def build_style(glass: bool) -> str:
    """Stylesheet: ``glass`` = Acrylic backdrop behind a translucent window, else the solid fallback."""
    if glass:
        return STYLE_TEMPLATE.format(root_bg=ROOT_GLASS, card_bg=CARD_GLASS)
    return STYLE_TEMPLATE.format(root_bg=ROOT_PLAIN, card_bg=CARD_PLAIN)
