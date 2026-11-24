"""Desktop presentation widgets. Chat markup is rendered without remote resources."""

import html
import re
from collections.abc import Callable
from typing import Any

import markdown
from markdown.treeprocessors import Treeprocessor
from pygments import highlight
from pygments.formatters import HtmlFormatter
from pygments.lexers import TextLexer, get_lexer_by_name
from pygments.util import ClassNotFound
from PySide6.QtCore import QSizeF, Qt, QTimer, QUrl, Signal
from PySide6.QtGui import QDesktopServices, QKeyEvent
from PySide6.QtWidgets import (
    QApplication,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QTextBrowser,
    QTextEdit,
    QVBoxLayout,
)

from infera_studio.conversations.records import Message

DARK = dict(
    base="#14171d",
    panel="#1b1f27",
    raised="#222731",
    input="#20252e",
    line="#333a47",
    text="#eef0f5",
    muted="#a9b3c4",
    accent="#96e5cb",
    selected="#263b39",
    ink="#11271f",
    code="#101319",
    danger="#ffb1ac",
)
LIGHT = dict(
    base="#faf9f5",
    panel="#f0f0eb",
    raised="#ffffff",
    input="#ffffff",
    line="#d9ded7",
    text="#202b28",
    muted="#546460",
    accent="#00674f",
    selected="#dceee5",
    ink="#ffffff",
    code="#edf1ed",
    danger="#9d272c",
)


def stylesheet(colors: dict[str, str]) -> str:
    c = colors
    return f"""
        QWidget {{
          background: {c["base"]}; color: {c["text"]};
          font-family: 'Inter', 'Ubuntu Sans', 'Sans Serif'; font-size: 14px;
        }}
        QWidget#sidebar {{ background: {c["panel"]}; border-right: 1px solid {c["line"]}; }}
        QLabel {{ background: transparent; }}
        QWidget#toolbar {{ border-bottom: 1px solid {c["line"]}; }}
        QDialog, QMessageBox {{ background: {c["base"]}; }}
        QLabel[role="title"] {{ font-size: 22px; font-weight: 700; }}
        QLabel[role="hero"] {{ font-size: 32px; font-weight: 700; }}
        QLabel[role="muted"] {{ color: {c["muted"]}; }}
        QLabel[role="status"] {{ color: {c["accent"]}; font-weight: 600; }}
        QLabel[role="error"] {{ color: {c["danger"]}; font-weight: 600; }}
        QLabel[role="speaker"] {{ color: {c["muted"]}; font-size: 12px; font-weight: 700; }}
        QPushButton, QComboBox, QSpinBox, QDoubleSpinBox, QLineEdit {{
          background: {c["input"]}; color: {c["text"]}; border: 1px solid {c["line"]};
          border-radius: 8px; padding: 6px 9px; min-height: 20px;
        }}
        QPushButton:hover, QComboBox:hover {{ background: {c["raised"]}; border-color: {c["muted"]}; }}
        QPushButton:focus, QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus, QLineEdit:focus
        {{ border: 1px solid {c["accent"]}; }}
        QPushButton:disabled, QComboBox:disabled {{ color: {c["muted"]}; border-color: {c["line"]}; }}
        QPushButton[role="primary"] {{
          background: {c["accent"]}; color: {c["ink"]}; font-weight: 700; padding: 8px 14px;
        }}
        QPushButton[role="primary"]:disabled {{ background: {c["raised"]}; color: {c["muted"]}; }}
        QPushButton[role="quiet"] {{ background: transparent; border: 1px solid transparent; color: {c["muted"]}; }}
        QPushButton[role="quiet"]:hover, QPushButton[role="quiet"]:focus {{
          color: {c["text"]}; border: 1px solid {c["line"]};
        }}
        QPushButton[role="danger"] {{ color: {c["danger"]}; }}
        QComboBox QAbstractItemView, QMenu {{
          background: {c["panel"]}; selection-background-color: {c["selected"]};
          selection-color: {c["text"]}; border: 1px solid {c["line"]};
        }}
        QMenu::item {{ padding: 7px 18px; }}
        QProgressBar {{ background: {c["line"]}; border: 0; border-radius: 3px; }}
        QProgressBar::chunk {{ background: {c["accent"]}; border-radius: 3px; }}
        QListWidget {{
          border: 0; background: transparent; outline: none;
          selection-background-color: {c["selected"]}; selection-color: {c["text"]};
        }}
        QListWidget::item {{ min-height: 32px; padding: 2px 8px; border-radius: 7px; }}
        QListWidget::item:hover {{ background: {c["raised"]}; }}
        QListWidget::item:selected {{ background: {c["selected"]}; }}
        QScrollArea, QTextBrowser {{ border: 0; background: transparent; }}
        QScrollBar:vertical {{ background: transparent; width: 9px; margin: 0; }}
        QScrollBar::handle:vertical {{ background: {c["line"]}; border-radius: 4px; min-height: 28px; }}
        QScrollBar::add-line, QScrollBar::sub-line {{ height: 0; width: 0; }}
        QTabWidget::pane {{ border: 1px solid {c["line"]}; border-radius: 8px; }}
        QTabBar::tab {{ background: {c["panel"]}; padding: 8px 15px; }}
        QTabBar::tab:selected {{ background: {c["selected"]}; }}
        QFrame[role="composer"], QFrame[role="user"], QFrame[role="notice"] {{
          background: {c["input"]}; border: 1px solid {c["line"]}; border-radius: 12px;
        }}
        QFrame[role="composer"] QWidget, QFrame[role="user"] QWidget, QFrame[role="notice"] QWidget {{
          background: transparent;
        }}
        QPlainTextEdit, QTextEdit#draft {{
          background: {c["input"]}; selection-background-color: {c["selected"]};
          selection-color: {c["text"]}; border: 1px solid {c["line"]}; border-radius: 8px;
        }}
        QTextEdit#draft {{ background: transparent; border: 0; padding: 1px; }}
        QToolTip {{ color: {c["text"]}; background: {c["raised"]}; border: 1px solid {c["line"]}; }}
    """


def label(text: str, role: str = "") -> QLabel:
    widget = QLabel(text)
    widget.setTextFormat(Qt.TextFormat.PlainText)
    widget.setProperty("role", role)
    return widget


def button(text: str, slot: Callable, role: str = "") -> QPushButton:
    widget = QPushButton(text)
    widget.setProperty("role", role)
    widget.clicked.connect(slot)
    return widget


class SafeLinks(Treeprocessor):
    def run(self, root: Any) -> Any:
        for parent in root.iter():
            for element in parent:
                if element.tag == "img":
                    element.tag = "span"
                    element.text = element.attrib.get("alt", "")
                    element.attrib.clear()
                elif element.tag == "a":
                    url = QUrl(element.get("href", ""))
                    if url.scheme() not in ("https", "http") or not url.host():
                        element.tag = "span"
                        element.attrib.clear()
        return root


CODE = re.compile(r'<pre><code(?: class="language-([^"]*)")?>(.*?)</code></pre>', re.DOTALL)


def render(text: str, colors: dict[str, str], dark: bool) -> tuple[str, list[str]]:
    engine = markdown.Markdown(extensions=["fenced_code", "tables", "sane_lists"])
    engine.preprocessors.deregister("html_block")
    engine.inlinePatterns.deregister("html")
    engine.inlinePatterns.deregister("entity")
    engine.treeprocessors.register(SafeLinks(engine), "safe_links", 1)
    raw = engine.convert(text)
    snippets: list[str] = []
    formatter = HtmlFormatter(noclasses=True, nowrap=True, style="monokai" if dark else "friendly")

    def code(match: re.Match[str]) -> str:
        language = html.unescape(match[1] or "")
        source = html.unescape(match[2])
        snippets.append(source)
        try:
            lexer = get_lexer_by_name(language) if language else TextLexer()
        except ClassNotFound:
            lexer = TextLexer()
        body = highlight(source, lexer, formatter)
        # Break long lines in the display while the copy action preserves original bytes.
        body = re.sub(
            r"[^\s<>]{100,}",
            lambda token: (
                token[0]
                if "=" in token[0] or "&" in token[0]
                else "\u200b".join(token[0][i : i + 64] for i in range(0, len(token[0]), 64))
            ),
            body,
        )
        return (
            f'<p><a href="infera-studio-copy:{len(snippets) - 1}">Copy code</a></p>'
            f'<pre style="background-color: {colors["code"]}; white-space: pre-wrap">{body}</pre>'
        )

    raw = CODE.sub(code, raw)
    return (
        f"<style>body {{color:{colors['text']}; font-size:14px;}} "
        f"a {{color:{colors['accent']};}} code, pre {{font-family:monospace;}} "
        f"td, th {{padding:4px; border:1px solid {colors['line']};}}</style>{raw}"
    ), snippets


class MarkdownView(QTextBrowser):
    def __init__(self, colors: dict[str, str], dark: bool):
        super().__init__()
        self.colors, self.dark = colors, dark
        self.snippets: list[str] = []
        self.setOpenLinks(False)
        self.setOpenExternalLinks(False)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setFocusPolicy(Qt.FocusPolicy.ClickFocus)
        self.document().setDocumentMargin(0)
        self.document().documentLayout().documentSizeChanged.connect(self._fit)
        self.anchorClicked.connect(self._anchor)

    def write(self, text: str, streaming: bool = False) -> None:
        if streaming:
            self.setPlainText(text)
        else:
            markup, self.snippets = render(text, self.colors, self.dark)
            self.setHtml(markup)
        self._fit()

    def _fit(self, _: QSizeF | None = None) -> None:
        height = max(20, int(self.document().size().height()) + 3)
        if self.height() != height:
            self.setFixedHeight(height)

    def loadResource(self, resource_type: int, name: QUrl) -> Any:
        return None

    def wheelEvent(self, event: Any) -> None:
        event.ignore()

    def _anchor(self, url: QUrl) -> None:
        if url.scheme() == "infera-studio-copy":
            index = url.path()
            if index.isdigit() and int(index) < len(self.snippets):
                QApplication.clipboard().setText(self.snippets[int(index)])
        elif url.scheme() in ("http", "https") and url.host():
            QDesktopServices.openUrl(url)


class MessageCard(QFrame):
    def __init__(
        self,
        message: Message,
        colors: dict[str, str],
        dark: bool,
        reasoning_open: bool,
        action: Callable[[Message], None],
    ):
        super().__init__()
        self.message, self.action = message, action
        self.setProperty("role", "user" if message.role == "user" else "assistant")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 8, 16, 10)
        layout.setSpacing(5)
        header = QHBoxLayout()
        header.addWidget(label("You" if message.role == "user" else "Assistant", "speaker"))
        header.addStretch()
        self.state = label("", "muted")
        header.addWidget(self.state)
        self.copy = button("Copy", self._copy, "quiet")
        self.copy.setToolTip("Copy message")
        header.addWidget(self.copy)
        self.edit = button(
            "Edit and resend" if message.role == "user" else "Regenerate",
            lambda: self.action(self.message),
            "quiet",
        )
        header.addWidget(self.edit)
        layout.addLayout(header)
        self.reasoning_button = button("Reasoning", self._toggle, "quiet")
        self.reasoning_button.setCheckable(True)
        self.reasoning_button.setChecked(reasoning_open)
        layout.addWidget(self.reasoning_button, alignment=Qt.AlignmentFlag.AlignLeft)
        self.reasoning = MarkdownView(colors, dark)
        layout.addWidget(self.reasoning)
        self.body = MarkdownView(colors, dark)
        layout.addWidget(self.body)
        self.update_message(message)

    def _toggle(self) -> None:
        self.reasoning.setVisible(self.reasoning_button.isChecked() and bool(self.message.reasoning))

    def _copy(self) -> None:
        QApplication.clipboard().setText(self.message.content)
        self.copy.setText("Copied")

        def reset() -> None:
            self.copy.setText("Copy")

        QTimer.singleShot(1200, self.copy, reset)

    def update_message(self, message: Message) -> None:
        self.message = message
        streaming = message.state == "streaming"
        text = message.content
        if not text.strip():
            if streaming:
                text = "Thinking…" if message.reasoning else "…"
            elif message.reasoning:
                text = "No final answer. Expand Reasoning to view what the model generated."
            else:
                text = "No answer was generated."
        self.body.write(text, streaming)
        self.reasoning_button.setVisible(bool(message.reasoning))
        self._toggle()
        if message.reasoning:
            self.reasoning.write(message.reasoning, streaming)
        self.state.setText(
            {
                "streaming": "Streaming",
                "stopped": "Stopped",
                "error": "Interrupted",
                "interrupted": "Interrupted",
                "no_answer": "Incomplete",
            }.get(message.state, "")
        )
        self.edit.setEnabled(not streaming)
        self.copy.setEnabled(bool(message.content.strip()))


class Composer(QTextEdit):
    submitted = Signal()

    def __init__(self) -> None:
        super().__init__()
        self.setObjectName("draft")
        self.setAccessibleName("Message")
        self.setAcceptRichText(False)
        self.setPlaceholderText("Message your local model")
        self.setFixedHeight(72)

    def keyPressEvent(self, event: QKeyEvent) -> None:
        if (
            event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter)
            and not event.modifiers() & Qt.KeyboardModifier.ShiftModifier
        ):
            self.submitted.emit()
            return
        super().keyPressEvent(event)
