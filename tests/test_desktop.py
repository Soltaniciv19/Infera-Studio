import os
from dataclasses import replace

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")
pytest.importorskip("markdown")
pytest.importorskip("pygments")

from PySide6.QtWidgets import QApplication

from infera_studio.desktop.qt.widgets import DARK, render
from infera_studio.desktop.qt.window import MainWindow
from infera_studio.persistence.sqlite import Store


def test_safe_markdown_and_exact_code_copy():
    markup, snippets = render(
        "<script>alert(1)</script>\n\n![hotlink](https://example.org/a.png)\n\n"
        "[local](file:///etc/passwd)\n\n```python\nprint('<hello>')\n```",
        DARK,
        True,
    )
    assert "&lt;script&gt;" in markup and "<script>" not in markup
    assert "<img" not in markup and "file:///" not in markup
    assert "infera-studio-copy:0" in markup and "color:" in markup
    assert snippets == ["print('<hello>')\n"]


def test_qt_chat_full_flow(tmp_path, mock_prefs, mock_model):
    from pathlib import Path

    from PySide6.QtCore import QEventLoop, QTimer

    app = QApplication.instance() or QApplication([])
    root = tmp_path / "data"
    root.mkdir()
    values = replace(
        mock_prefs, model_dirs=(str(Path(mock_model.path).parent),), backend="CPU", setup_complete=True
    )
    values.save(root / "settings")
    store = Store(root / "history.db")
    window = MainWindow(store, root / "settings")
    app.setStyle("Fusion")
    window.show()

    def wait(predicate):
        loop = QEventLoop()
        timer = QTimer(interval=10)
        timer.timeout.connect(lambda: loop.quit() if predicate() else None)
        timer.start()
        timeout = QTimer(singleShot=True, interval=8000)
        timeout.timeout.connect(loop.quit)
        timeout.start()
        if not predicate():
            loop.exec()
        timer.stop()
        timeout.stop()
        assert predicate()

    try:
        wait(lambda: window.initialized and bool(window.models))
        assert window.selector.parentWidget() is window.composer.parentWidget()
        assert window.selector.y() > window.composer.geometry().bottom()
        assert window.selector.x() < window.send_button.x()
        index = window.selector.findData(mock_model.path)
        assert index > 0
        window.selector.setCurrentIndex(index)
        window.selector.activated.emit(index)
        wait(lambda: window.status.text() == "Ready" and window.operation is None)
        window.composer.setPlainText("Hello through Qt")
        window.send_button.click()
        wait(lambda: len(window.messages) == 2 and window.operation is None)
        assert "20.0 generation tok/s" in window.metrics.text()
        chat, turns = store.get(window.chat.id)
        assert chat.title == "Hello through Qt" and chat.model == mock_model.path
        assert [turn.content for turn in turns] == ["Hello through Qt", "Hello world"]
        assert turns[-1].reasoning == "private"
        window.composer.setPlainText("slow")
        window.send_button.click()
        wait(lambda: len(window.messages) == 4 and window.stop.isVisible())
        window.stop.click()
        wait(lambda: window.operation is None)
        assert store.get(chat.id)[1][-1].state == "stopped"
        window._message_action(window.messages[1])
        wait(
            lambda: (
                window.chat is not None
                and window.chat.id != chat.id
                and len(window.messages) == 2
                and window.operation is None
            )
        )
        assert len(store.get(chat.id)[1]) == 4
        assert window.messages[-1].content == "Hello world"

        branch_id = window.chat.id
        window._configure(mock_model.path, replace(window.chat.settings, thinking="On"))
        wait(
            lambda: (
                window.chat is not None
                and window.chat.settings.thinking == "On"
                and window.operation is None
                and window.status.text() == "Ready"
            )
        )
        window.composer.setPlainText("reasoning-only")
        window.send_button.click()
        wait(lambda: len(window.messages) == 4 and window.operation is None)
        reply = window.messages[-1]
        card = window.cards[reply.id]
        assert reply.state == "no_answer"
        assert reply.content == "" and reply.reasoning == "Still thinking"
        assert "No final answer" in card.body.toPlainText()
        wait(card.reasoning_button.isVisible)
        assert not card.reasoning.isVisible()
        card.reasoning_button.click()
        wait(card.reasoning.isVisible)
        assert "output token limit while thinking" in window.notice_text.text()

        window._configure(mock_model.path, replace(window.chat.settings, thinking="Automatic"))
        wait(
            lambda: (
                window.chat is not None
                and window.chat.settings.thinking == "Automatic"
                and window.operation is None
                and window.status.text() == "Ready"
            )
        )
        window.cards[reply.id].edit.click()
        wait(
            lambda: (
                window.chat is not None
                and window.chat.id != branch_id
                and len(window.messages) == 4
                and window.operation is None
            )
        )
        assert window.messages[-1].content == "Hello world"
        assert "Hello world" in window.cards[window.messages[-1].id].body.toPlainText()
    finally:
        window.close()
        wait(lambda: window.closed)
    assert window.runtime.server.child is None
