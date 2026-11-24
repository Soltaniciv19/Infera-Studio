"""Infera Studio desktop window and background operation coordination."""

import logging
from dataclasses import replace
from pathlib import Path
from typing import Any

from PySide6.QtCore import QSignalBlocker, Qt, QTimer
from PySide6.QtGui import QCloseEvent, QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QDialog,
    QFrame,
    QHBoxLayout,
    QInputDialog,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMenu,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from infera_studio.application.runtime import BudgetExceeded, Generation, Runtime
from infera_studio.catalog.gguf import Model, scan
from infera_studio.config.preferences import Preferences, default_model_dirs
from infera_studio.config.settings import InferenceSettings
from infera_studio.conversations.metrics import Metrics
from infera_studio.conversations.records import Chat, Message, now
from infera_studio.host.hardware import GPU, amd_gpus
from infera_studio.inference.client import Budget, Cancellation, Cancelled
from infera_studio.inference.memory import recommend
from infera_studio.inference.server import ServerInfo, detect
from infera_studio.persistence.sqlite import Store

from .dialogs import ContextDialog, ModelDialog, PreferencesDialog, SetupWizard
from .widgets import DARK, LIGHT, Composer, MessageCard, button, label, stylesheet
from .workers import Workers

LOG = logging.getLogger(__name__)
USER_ROLE = Qt.ItemDataRole.UserRole


class MainWindow(QMainWindow):
    def __init__(self, store: Store, config_path: Path):
        super().__init__()
        self.setWindowTitle("Infera Studio")
        self.resize(1180, 790)
        self.setMinimumSize(820, 570)
        self.store, self.config_path = store, config_path
        self.runtime, self.workers = Runtime(store), Workers()
        self.workers.event.connect(self._event)
        self.prefs = Preferences(model_dirs=())
        self.models: dict[str, Model] = {}
        self.discovered: dict[str, Path | None] = {}
        self.chat: Chat | None = None
        self.messages: list[Message] = []
        self.cards: dict[int, MessageCard] = {}
        self.drafts: dict[int, str] = {}
        self.info: ServerInfo | None = None
        self.gpu: GPU | None = None
        self.budget: Budget | None = None
        self.last_tokens: int | None = None
        self.operation: Cancellation | None = None
        self.count_cancel: Cancellation | None = None
        self.revision = 0
        self.stream_id = 0
        self.pending_edit: int | None = None
        self.scan_errors: list[str] = []
        self.scanning = False
        self.initialized = False
        self.closing = False
        self.closed = False
        self.following = True
        self.list_revision = 0
        self.health_pending = False
        self._build()
        self._theme()
        self._status("Stopped", "Opening local data")
        self.search_delay = QTimer(self, singleShot=True, interval=210)
        self.search_delay.timeout.connect(self._refresh_history)
        self.search.textChanged.connect(lambda: self.search_delay.start())
        self.count_delay = QTimer(self, singleShot=True, interval=330)
        self.count_delay.timeout.connect(self._count)
        self.composer.textChanged.connect(self._draft_changed)
        self.health_timer = QTimer(self, interval=4500)
        self.health_timer.timeout.connect(self._health)
        self.health_timer.start()
        QTimer.singleShot(0, self._initialize)

    def _build(self) -> None:
        central = QWidget()
        shell = QHBoxLayout(central)
        shell.setContentsMargins(0, 0, 0, 0)
        shell.setSpacing(0)
        sidebar = QWidget()
        sidebar.setObjectName("sidebar")
        sidebar.setFixedWidth(242)
        side = QVBoxLayout(sidebar)
        side.setContentsMargins(17, 20, 17, 17)
        side.setSpacing(12)
        side.addWidget(label("Infera Studio", "title"))
        self.new = button("＋  New chat", self.new_chat, "primary")
        side.addWidget(self.new)
        self.search = QLineEdit()
        self.search.setPlaceholderText("Search conversations")
        self.search.setAccessibleName("Search conversations")
        side.addWidget(self.search)
        self.history = QListWidget()
        self.history.setAccessibleName("Conversations")
        self.history.itemClicked.connect(self._select_chat)
        self.history.itemActivated.connect(self._select_chat)
        self.history.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.history.customContextMenuRequested.connect(self._history_menu)
        side.addWidget(self.history, 1)
        self.chat_actions = button("Conversation actions…", self._active_menu, "quiet")
        side.addWidget(self.chat_actions)
        self.settings = button("Settings", self.preferences, "quiet")
        side.addWidget(self.settings)
        shell.addWidget(sidebar)

        pane = QWidget()
        column = QVBoxLayout(pane)
        column.setContentsMargins(0, 0, 0, 0)
        column.setSpacing(0)
        toolbar = QWidget()
        toolbar.setObjectName("toolbar")
        row = QHBoxLayout(toolbar)
        row.setContentsMargins(18, 16, 18, 16)
        row.setSpacing(9)
        row.addStretch(1)
        self.backend = label("Vulkan", "muted")
        row.addWidget(self.backend)
        self.status = label("Stopped", "status")
        row.addWidget(self.status)
        self.reload = button("Server…", self._server_menu, "quiet")
        row.addWidget(self.reload)
        self.context = button("Context", self.context_dialog)
        row.addWidget(self.context)
        self.model_config = button("Model settings", self.model_settings, "quiet")
        row.addWidget(self.model_config)
        column.addWidget(toolbar)

        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        viewport = QWidget()
        viewport_layout = QHBoxLayout(viewport)
        viewport_layout.setContentsMargins(16, 10, 16, 4)
        viewport_layout.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        transcript = QWidget()
        transcript.setMaximumWidth(830)
        self.transcript = QVBoxLayout(transcript)
        self.transcript.setContentsMargins(5, 8, 5, 15)
        self.transcript.setSpacing(13)
        viewport_layout.addWidget(transcript, 1)
        self.scroll.setWidget(viewport)
        self.scroll.verticalScrollBar().valueChanged.connect(self._scroll_changed)
        self.scroll.verticalScrollBar().rangeChanged.connect(self._scroll_range)
        column.addWidget(self.scroll, 1)

        bottom = QWidget()
        bottom_row = QHBoxLayout(bottom)
        bottom_row.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        bottom_row.setContentsMargins(19, 4, 19, 13)
        composer_group = QWidget()
        composer_group.setMaximumWidth(825)
        controls = QVBoxLayout(composer_group)
        controls.setContentsMargins(0, 0, 0, 0)
        controls.setSpacing(7)
        self.notice = QFrame()
        self.notice.setProperty("role", "notice")
        notice_layout = QHBoxLayout(self.notice)
        notice_layout.setContentsMargins(10, 5, 8, 5)
        self.notice_text = label("", "muted")
        self.notice_text.setWordWrap(True)
        self.notice_text.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        notice_layout.addWidget(self.notice_text, 1)
        self.notice_action = button("Details", self._notice_action, "quiet")
        notice_layout.addWidget(self.notice_action)
        self.notice.hide()
        controls.addWidget(self.notice)
        self.loading = QProgressBar()
        self.loading.setRange(0, 0)
        self.loading.setTextVisible(False)
        self.loading.setFixedHeight(4)
        self.loading.hide()
        controls.addWidget(self.loading)
        composer_frame = QFrame()
        composer_frame.setProperty("role", "composer")
        layout = QVBoxLayout(composer_frame)
        layout.setContentsMargins(12, 7, 8, 7)
        layout.setSpacing(2)
        self.composer = Composer()
        self.composer.submitted.connect(self.send)
        layout.addWidget(self.composer)
        actions = QHBoxLayout()
        actions.setSpacing(8)
        self.selector = QComboBox()
        self.selector.setAccessibleName("Local model")
        self.selector.setMinimumWidth(165)
        self.selector.setMaximumWidth(300)
        self.selector.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.selector.setMinimumContentsLength(16)
        self.selector.activated.connect(self._select_model)
        actions.addWidget(self.selector, 1)
        self.detail = label("", "muted")
        self.detail.setWordWrap(True)
        actions.addWidget(self.detail, 2)
        self.stop = button("Stop", self._stop, "danger")
        actions.addWidget(self.stop)
        self.send_button = button("Send  ↑", self.send, "primary")
        actions.addWidget(self.send_button)
        layout.addLayout(actions)
        controls.addWidget(composer_frame)
        self.metrics = label("", "muted")
        self.metrics.setWordWrap(True)
        controls.addWidget(self.metrics)
        bottom_row.addWidget(composer_group, 1)
        column.addWidget(bottom)
        shell.addWidget(pane, 1)
        self.setCentralWidget(central)
        self.new_shortcut = QShortcut(QKeySequence("Ctrl+N"), self)
        self.new_shortcut.activated.connect(self.new_chat)
        self.search_shortcut = QShortcut(QKeySequence("Ctrl+K"), self)
        self.search_shortcut.activated.connect(self.search.setFocus)
        self.stop_shortcut = QShortcut(QKeySequence("Escape"), self)
        self.stop_shortcut.activated.connect(self._stop)
        self.notice_kind = ""
        self._draw_chat()
        self._choices()
        self._controls()

    def _theme(self) -> None:
        self.colors = DARK if self.prefs.theme == "Dark" else LIGHT
        app = QApplication.instance()
        app.setStyleSheet(stylesheet(self.colors))

    def _initialize(self) -> None:
        def setup() -> tuple[Preferences, str, dict[str, Path | None], list[Chat], dict[str, Model]]:
            self.store.initialize()
            note = ""
            try:
                prefs = Preferences.load(self.config_path)
            except (OSError, ValueError) as exc:
                prefs = Preferences(model_dirs=default_model_dirs())
                note = f"Settings could not be read. Open Settings to replace the file. {exc}"
            return prefs, note, detect(prefs), self.store.search(), self.store.catalog()

        self.workers.submit(self.workers.disk, setup, self._initialized, self._startup_failed)

    def _initialized(
        self, result: tuple[Preferences, str, dict[str, Path | None], list[Chat], dict[str, Model]]
    ) -> None:
        if self.closing:
            return
        self.prefs, note, self.discovered, chats, self.models = result
        self.initialized = True
        self._theme()
        self._status("Stopped", "")
        self._choices()
        self._populate_history(chats)
        if not self.prefs.setup_complete and not note:
            wizard = SetupWizard(self, self.prefs, self.discovered)
            if wizard.exec() == QDialog.DialogCode.Accepted:
                self._apply_preferences(wizard.values(), chats[0].id if chats else None)
                return
        if note:
            self._notice(note, "settings")
        self.rescan(open_chat=chats[0].id if chats else None)

    def _startup_failed(self, exc: Exception) -> None:
        if not self.closing:
            self._status("Error", "Could not open Infera Studio data")
            self._notice(str(exc), "diagnostics")
            QMessageBox.critical(self, "Could not open Infera Studio data", str(exc))

    def _notice(self, text: str = "", kind: str = "") -> None:
        self.notice_kind = kind
        self.notice_text.setText(text)
        self.notice.setVisible(bool(text))
        self.notice_action.setText(
            {
                "settings": "Settings",
                "context": "Context options",
                "diagnostics": "Diagnostics",
                "models": "Model library",
                "model-settings": "Model settings",
            }.get(kind, "Details")
        )
        self.notice_action.setVisible(bool(kind))

    def _notice_action(self) -> None:
        actions = {
            "settings": self.preferences,
            "context": self.context_dialog,
            "diagnostics": self.diagnostics,
            "models": self.model_library,
            "model-settings": self.model_settings,
        }
        if action := actions.get(self.notice_kind):
            action()

    def _error(self, exc: Exception, kind: str = "diagnostics") -> None:
        if not self.closing:
            self._notice(str(exc), kind)

    def _status(self, state: str, detail: str) -> None:
        self.status.setText(state)
        self.status.setProperty("role", "error" if state == "Error" else "status")
        self.status.style().unpolish(self.status)
        self.status.style().polish(self.status)
        self.detail.setText(detail)
        self.loading.setVisible(state == "Loading")
        self._controls()

    def _controls(self) -> None:
        enabled = self.initialized and not self.closing
        self.new.setEnabled(enabled)
        self.search.setEnabled(enabled)
        self.settings.setEnabled(enabled)
        self.history.setEnabled(enabled)
        self.selector.setEnabled(enabled)
        self.chat_actions.setEnabled(enabled and self.chat is not None)
        self.reload.setEnabled(enabled)
        self.context.setEnabled(enabled and self.chat is not None and self.operation is None)
        self.model_config.setEnabled(enabled and self.chat is not None and self.operation is None)
        self.composer.setReadOnly(not enabled or self.operation is not None)
        self.stop.setVisible(self.operation is not None and enabled)
        self.send_button.setEnabled(
            enabled
            and self.chat is not None
            and self._selected() is not None
            and self.operation is None
            and bool(self.composer.toPlainText().strip())
        )

    def _refresh_history(self) -> None:
        if not self.initialized or self.closing:
            return
        self.list_revision += 1
        stamp = self.list_revision
        query = self.search.text().strip()
        self.workers.submit(
            self.workers.disk,
            lambda: self.store.search(query),
            lambda found: self._populate_history(found) if stamp == self.list_revision else None,
            self._error,
        )

    def _populate_history(self, chats: list[Chat]) -> None:
        if self.closing:
            return
        self.history.clear()
        active = self.chat.id if self.chat else None
        for chat in chats:
            item = QListWidgetItem(chat.title)
            item.setData(USER_ROLE, chat.id)
            item.setToolTip(chat.title)
            self.history.addItem(item)
            if active == chat.id:
                self.history.setCurrentItem(item)

    def _select_chat(self, item: QListWidgetItem) -> None:
        chat_id = item.data(USER_ROLE)
        if self.chat is None or chat_id != self.chat.id:
            self._open(chat_id)

    def _stash_draft(self) -> None:
        if self.chat:
            draft = self.composer.toPlainText()
            if draft.strip():
                self.drafts[self.chat.id] = draft
            else:
                self.drafts.pop(self.chat.id, None)

    def _invalidate(self) -> int:
        if self.operation:
            self.operation.cancel()
        if self.count_cancel:
            self.count_cancel.cancel()
            self.count_cancel = None
        self.operation = None
        self.revision += 1
        return self.revision

    def _open(self, chat_id: int, regenerate: bool = False) -> None:
        self._stash_draft()
        stamp = self._invalidate()
        self.chat, self.messages, self.info, self.budget = None, [], None, None
        self.metrics.setText("")
        self.last_tokens = None
        self.context.setText("Context")
        self._notice()
        self._status("Stopped", "Opening conversation")
        self._draw_chat()

        def opened(value: tuple[Chat, list[Message]]) -> None:
            if stamp != self.revision or self.closing:
                return
            self.chat, self.messages = value
            self.composer.setPlainText(self.drafts.get(chat_id, ""))
            self.composer.setFocus()
            self._choices()
            self._draw_chat()
            self._refresh_history()
            if regenerate:
                self._generate(None)
            else:
                self._load()

        self.workers.submit(self.workers.inference, lambda: self.store.get(chat_id), opened, self._error)

    def new_chat(self) -> None:
        if not self.initialized or self.closing:
            return
        model = self._selected() or next((item for item in self.models.values() if not item.issue), None)
        settings = self.chat.settings if self.chat else InferenceSettings(backend=self.prefs.backend)
        self._new(model.path if model else "", replace(settings, system=""))

    def _new(self, model: str, settings: InferenceSettings) -> None:
        stamp = self.revision
        self.workers.submit(
            self.workers.disk,
            lambda: self.store.create(model, settings),
            lambda chat_id: self._open(chat_id) if stamp == self.revision else self._refresh_history(),
            self._error,
        )

    def _history_menu(self, pos: Any) -> None:
        if item := self.history.itemAt(pos):
            self._conversation_menu(item.data(USER_ROLE), item.text(), self.history.mapToGlobal(pos))

    def _active_menu(self) -> None:
        if self.chat:
            self._conversation_menu(
                self.chat.id,
                self.chat.title,
                self.chat_actions.mapToGlobal(self.chat_actions.rect().bottomLeft()),
            )

    def _conversation_menu(self, chat_id: int, title: str, position: Any) -> None:
        menu = QMenu(self)
        menu.addAction("Rename…", lambda: self._rename(chat_id, title))
        menu.addAction("Delete…", lambda: self._delete(chat_id, title))
        menu.addSeparator()
        menu.addAction("Delete all conversations…", lambda: self._delete(None, ""))
        menu.exec(position)

    def _rename(self, chat_id: int, title: str) -> None:
        name, ok = QInputDialog.getText(self, "Rename conversation", "Conversation title", text=title)
        if ok and name.strip():
            name = name.strip()

            def renamed(_: Any) -> None:
                if self.chat and self.chat.id == chat_id:
                    self.chat = replace(self.chat, title=name)
                self._refresh_history()

            self.workers.submit(
                self.workers.disk, lambda: self.store.rename(chat_id, name), renamed, self._error
            )

    def _delete(self, chat_id: int | None, title: str) -> None:
        detail = (
            f"Delete “{title}” and its messages?"
            if chat_id is not None
            else "Delete every conversation and its messages?"
        )
        if (
            QMessageBox.question(
                self,
                "Delete conversations",
                detail,
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
                QMessageBox.StandardButton.Cancel,
            )
            != QMessageBox.StandardButton.Yes
        ):
            return
        selected = chat_id is None or (self.chat is not None and self.chat.id == chat_id)
        if selected:
            self._invalidate()
            self.chat = None
            self.messages = []
            self.info, self.budget = None, None
            self.composer.clear()
            self._draw_chat()
            self._choices()
            self._stop_server()
        if chat_id is None:
            self.drafts.clear()
        else:
            self.drafts.pop(chat_id, None)
        self.workers.submit(
            self.workers.disk,
            lambda: self.store.delete(chat_id),
            lambda _: self._refresh_history(),
            self._error,
        )

    def _selected(self) -> Model | None:
        if self.chat:
            model = self.models.get(self.chat.model)
            if model and not model.issue:
                return model
        return None

    def _choices(self) -> None:
        with QSignalBlocker(self.selector):
            self.selector.clear()
            path = self.chat.model if self.chat else ""
            selected = self.models.get(path)
            heading = "Select a model"
            if path and (selected is None or selected.issue):
                heading = f"Unavailable · {selected.name if selected else Path(path).stem}"
            self.selector.addItem(heading, "")
            for model in sorted(self.models.values(), key=lambda m: (m.name.casefold(), m.path)):
                if model.issue:
                    continue
                self.selector.addItem(model.name, model.path)
                index = self.selector.count() - 1
                self.selector.setItemData(
                    index, f"{model.path}\n{model.quant} · {model.architecture}", Qt.ItemDataRole.ToolTipRole
                )
                if model.path == path:
                    self.selector.setCurrentIndex(index)
            self.selector.insertSeparator(self.selector.count())
            self.selector.addItem("Model library…", "__library")
            self.selector.addItem("Rescan model folders", "__scan")
        self.backend.setText(self.chat.settings.backend if self.chat else self.prefs.backend)
        self._controls()

    def _select_model(self, index: int) -> None:
        path = self.selector.itemData(index)
        if path == "__library":
            self._choices()
            self.model_library()
        elif path == "__scan":
            self._choices()
            self.rescan()
        elif path and self.chat and path != self.chat.model:
            self._configure(path, self.chat.settings)
        elif path and self.chat is None:
            self._new(path, InferenceSettings(backend=self.prefs.backend))
        else:
            self._choices()

    def _configure(self, model: str, settings: InferenceSettings) -> None:
        if self.chat is None:
            return
        self._stash_draft()
        stamp = self._invalidate()
        chat_id = self.chat.id
        self.info, self.budget = None, None
        self._show_budget(None)
        self._status("Stopped", "Saving model configuration")
        self.workers.submit(
            self.workers.disk,
            lambda: self.store.configure(chat_id, model, settings),
            lambda _: self._open(chat_id) if stamp == self.revision else None,
            self._error,
        )

    def _draw_chat(self) -> None:
        while self.transcript.count():
            widget = self.transcript.takeAt(0).widget()
            if widget:
                widget.deleteLater()
        self.cards.clear()
        self.following = True
        if not self.messages:
            self.transcript.addStretch(2)
            title = label("Start a conversation" if self.chat else "Your local AI chat", "hero")
            title.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.transcript.addWidget(title)
            subtitle = label(
                "Send a message to the selected model."
                if self._selected()
                else "Choose a GGUF model to get started.",
                "muted",
            )
            subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.transcript.addWidget(subtitle)
            self.transcript.addStretch(3)
        else:
            for message in self.messages:
                self._add_message(message)
            self.transcript.addStretch()
        self._scroll_bottom(True)

    def _add_message(self, message: Message) -> None:
        card = MessageCard(
            message, self.colors, self.prefs.theme == "Dark", self.prefs.show_reasoning, self._message_action
        )
        self.cards[message.id] = card
        self.transcript.addWidget(card)

    def _scroll_changed(self, value: int) -> None:
        self.following = self.scroll.verticalScrollBar().maximum() - value < 50

    def _scroll_range(self, _minimum: int, _maximum: int) -> None:
        self._scroll_bottom()

    def _scroll_bottom(self, force: bool = False) -> None:
        if force or self.following:
            bar = self.scroll.verticalScrollBar()
            QTimer.singleShot(0, self, lambda: bar.setValue(bar.maximum()))

    def _message_action(self, message: Message) -> None:
        if not self.chat or self.operation:
            return
        if message.role == "user":
            text, accepted = QInputDialog.getMultiLineText(
                self, "Edit and resend", "Message", message.content
            )
            if not accepted or not text.strip():
                return
        else:
            text = ""
            if not any(turn.role == "user" and turn.id < message.id for turn in self.messages):
                return
        chat_id, stamp = self.chat.id, self.revision

        def forked(branch: int) -> None:
            if stamp != self.revision:
                self._refresh_history()
                return
            if text:
                self.drafts[branch] = text
            self.pending_edit = branch if text else None
            self._open(branch, regenerate=not text)

        self.workers.submit(
            self.workers.disk, lambda: self.store.fork(chat_id, message.id), forked, self._error
        )

    def _status_event(self, stamp: int) -> Any:
        return lambda state, detail: self.workers.event.emit("status", (stamp, state, detail))

    def _begin(self, state: str, detail: str) -> tuple[int, Cancellation]:
        stamp = self._invalidate()
        self.operation = cancel = Cancellation()
        self._status(state, detail)
        return stamp, cancel

    def _stop(self) -> None:
        if self.operation:
            self.operation.cancel()
            self.detail.setText("Stopping…")

    def _stop_server(self) -> None:
        stamp = self.revision
        self.workers.submit(
            self.workers.inference,
            self.runtime.server.stop,
            lambda _: self._status("Stopped", "") if stamp == self.revision else None,
            self._error,
        )

    def _load(self) -> None:
        if self.closing or self.chat is None:
            return
        model, settings = self._selected(), self.chat.settings
        if model is None:
            self._stop_server()
            self._status("Stopped", "")
            self._notice(
                "The selected model is unavailable. Choose a replacement to keep chatting."
                if self.chat.model
                else "Select a model to start chatting.",
                "models",
            )
            return
        prefs = self.prefs
        stamp, cancel = self._begin("Loading", "Preparing the local model")
        self._notice()

        def loaded(info: ServerInfo) -> None:
            if self.closing or stamp != self.revision:
                return
            self.operation = None
            self.info = info
            self._status("Ready", "")
            self.count_delay.start()
            if self.chat and self.chat.id == self.pending_edit:
                self.pending_edit = None
                self.send()

        self.workers.submit(
            self.workers.inference,
            lambda: self.runtime.server.load(model, settings, prefs, cancel, self._status_event(stamp)),
            loaded,
            lambda exc: self._operation_error(stamp, exc),
        )

    def _operation_error(self, stamp: int, exc: Exception) -> None:
        if self.closing or stamp != self.revision:
            return
        self.operation = None
        if isinstance(exc, BudgetExceeded):
            self._show_budget(exc.budget)
            self._status("Ready", "Context limit reached")
            self._notice(str(exc), "context")
        elif isinstance(exc, Cancelled):
            self._status("Ready" if self.info else "Stopped", "Stopped")
        else:
            self.info = None
            self._status("Error", "Inference needs attention")
            self._notice(str(exc), "diagnostics")

    def _event(self, kind: str, payload: tuple) -> None:
        if self.closing or payload[0] != self.revision:
            return
        _, *values = payload
        if kind == "status":
            self._status(*values)
        elif kind == "ready":
            self.info = values[0]
        elif kind == "budget":
            self._show_budget(values[0])
        elif kind == "started":
            self.stream_id, draft = values
            if draft is not None and self.chat:
                self.drafts.pop(self.chat.id, None)
                self.composer.clear()
                self.messages.append(Message(0, "user", draft, now(), "", "complete"))
            self.messages.append(Message(self.stream_id, "assistant", "", now(), "", "streaming"))
            self._draw_chat()
        elif kind == "delta" and self.stream_id in self.cards:
            text, reasoning = values
            card = self.cards[self.stream_id]
            card.update_message(replace(card.message, content=text, reasoning=reasoning))
            self._scroll_bottom()

    def send(self) -> None:
        draft = self.composer.toPlainText()
        if draft.strip() and self.operation is None:
            self._generate(draft)

    def _generate(self, draft: str | None) -> None:
        chat, model = self.chat, self._selected()
        if chat is None or model is None or self.operation is not None or self.closing:
            return
        messages, prefs = self.messages.copy(), self.prefs
        self.metrics.clear()
        self._notice()
        stamp, cancel = self._begin("Generating", "Checking context")

        def notify(kind: str, *args: Any) -> None:
            self.workers.event.emit(kind, (stamp, *args))

        def generate() -> Generation:
            return self.runtime.generate(
                model,
                prefs,
                chat,
                messages,
                draft,
                cancel,
                self._status_event(stamp),
                lambda info: notify("ready", info),
                lambda budget: notify("budget", budget),
                lambda msg_id: notify("started", msg_id, draft),
                lambda text, thoughts: notify("delta", text, thoughts),
            )

        self.workers.submit(
            self.workers.inference,
            generate,
            lambda result: self._finished(stamp, chat.id, result),
            lambda exc: self._operation_error(stamp, exc),
        )

    def _finished(self, stamp: int, chat_id: int, result: Generation) -> None:
        if self.closing:
            return
        if self.revision != stamp:
            self._refresh_history()
            return
        self.detail.setText("Saving response…")
        if result.error:
            self._notice(result.error, "diagnostics")
        elif result.notice:
            self._notice(result.notice, "model-settings")
        metrics = result.metrics
        self.last_tokens = (
            metrics.prompt_tokens + metrics.generated_tokens
            if metrics.prompt_tokens is not None and metrics.generated_tokens is not None
            else None
        )
        self._show_metrics(metrics)
        self._show_budget(self.budget)
        self.workers.submit(
            self.workers.disk,
            lambda: self.store.get(chat_id),
            lambda value: self._refreshed(value, stamp, result),
            lambda exc: self._operation_error(stamp, exc),
        )

    def _refreshed(self, value: tuple[Chat, list[Message]], stamp: int, result: Generation) -> None:
        if stamp == self.revision and not self.closing:
            self.operation = None
            self._status(
                "Ready" if result.state != "error" else "Error",
                "Response stopped" if result.state == "stopped" else "",
            )
            following = self.following
            scroll = self.scroll.verticalScrollBar().value()
            self.chat, self.messages = value
            self._draw_chat()
            if not following:
                self.following = False
                QTimer.singleShot(0, self, lambda: self.scroll.verticalScrollBar().setValue(scroll))
            self._refresh_history()
            self.count_delay.start()

    def _show_metrics(self, metrics: Metrics) -> None:
        parts = []
        for value, pattern in (
            (metrics.prompt_per_second, "{:.1f} prompt tok/s"),
            (metrics.generation_per_second, "{:.1f} generation tok/s"),
            (metrics.first_token_seconds, "{:.2f}s to first token"),
            (metrics.generated_tokens, "{:,} generated"),
            (metrics.prompt_tokens, "{:,} prompt tokens"),
            (metrics.seconds, "{:.2f}s total"),
        ):
            if value is not None:
                parts.append(pattern.format(value))
        self.metrics.setText("  ·  ".join(parts))

    def _draft_changed(self) -> None:
        if self.count_cancel:
            self.count_cancel.cancel()
            self.count_cancel = None
        self.count_delay.start()
        self._controls()

    def _count(self) -> None:
        if self.closing or self.operation or self.chat is None or self.info is None:
            return
        chat, info, stamp = self.chat, self.info, self.revision
        messages, draft = self.messages.copy(), self.composer.toPlainText()
        if not draft.strip():
            draft = ""
        if self.count_cancel:
            self.count_cancel.cancel()
        self.count_cancel = cancel = Cancellation()

        def counted(budget: Budget) -> None:
            if stamp == self.revision and not cancel.cancelled and not self.closing:
                self._show_budget(budget)
                if budget.remaining >= 0 and self.notice_kind == "context":
                    self._notice()
                elif budget.remaining < 0:
                    self._notice(
                        f"Context needs {-budget.remaining:,} more tokens for this request.", "context"
                    )

        def failed(exc: Exception) -> None:
            if stamp == self.revision and not cancel.cancelled and not self.closing and not self.notice_kind:
                self._notice(str(exc), "context")

        self.workers.submit(
            self.workers.inference,
            lambda: self.runtime.budget(info, chat, messages, draft, cancel),
            counted,
            failed,
        )

    def _show_budget(self, budget: Budget | None) -> None:
        self.budget = budget
        if budget is None:
            self.context.setText("Context")
            self.context.setToolTip("")
            return
        prefix = "~" if budget.estimated else ""
        self.context.setText(f"{prefix}{budget.prompt:,} / {budget.context:,}")
        consumed = (
            f" Last inference used {self.last_tokens:,} prompt + output tokens."
            if self.last_tokens is not None
            else ""
        )
        self.context.setToolTip(
            f"{budget.reserved:,} reserved output · {budget.remaining:,} remaining."
            f"{consumed} Open for token and GPU details."
        )

    def model_settings(self) -> None:
        if self.chat is None or self.operation:
            return
        dialog = ModelDialog(self, self.chat.settings, self._selected(), self.info, self.gpu)
        if dialog.exec() == QDialog.DialogCode.Accepted and dialog.config != self.chat.settings:
            self._configure(self.chat.model, dialog.config)

    def context_dialog(self) -> None:
        if self.chat is None or self.operation:
            return
        chat, info = self.chat, self.info
        model = self._selected()
        plan = info.plan if info else (recommend(model, chat.settings, self.gpu) if model else None)
        dialog = ContextDialog(self, chat, self.messages, self.budget, plan, self.gpu, info)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        if dialog.action == "settings":
            self.model_settings()
        elif dialog.action == "new":
            self.new_chat()
        elif dialog.action == "cut" and dialog.start != chat.active_from:
            self._stash_draft()
            self.workers.submit(
                self.workers.disk,
                lambda: self.store.set_context(chat.id, dialog.start),
                lambda _: self._open(chat.id) if self.chat and self.chat.id == chat.id else None,
                self._error,
            )
        elif dialog.action == "summarize" and info:
            stamp, cancel = self._begin("Generating", "Summarizing earlier turns")
            messages = self.messages.copy()
            self.workers.submit(
                self.workers.inference,
                lambda: self.runtime.summarize(info, chat, messages, dialog.start, cancel),
                lambda _: self._open(chat.id) if self.revision == stamp else None,
                lambda exc: self._operation_error(stamp, exc),
            )

    def preferences(self) -> None:
        if not self.initialized or self.closing:
            return
        dialog = PreferencesDialog(self, self.prefs, self.discovered)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            self._apply_preferences(dialog.values())

    def _apply_preferences(self, prefs: Preferences, open_chat: int | None = None) -> None:
        previous = self.prefs

        def save() -> dict[str, Path | None]:
            prefs.save(self.config_path)
            return detect(prefs)

        def saved(discovered: dict[str, Path | None]) -> None:
            if self.closing:
                return
            self.prefs, self.discovered = prefs, discovered
            self._theme()
            if previous.theme != prefs.theme or previous.show_reasoning != prefs.show_reasoning:
                self._draw_chat()
            if self.chat and previous.binaries != prefs.binaries:
                self._open(self.chat.id)
            if previous.model_dirs != prefs.model_dirs or open_chat is not None or not self.models:
                self.rescan(open_chat=open_chat)
            else:
                self._choices()

        self.workers.submit(self.workers.disk, save, saved, self._error)

    def rescan(self, open_chat: int | None = None) -> None:
        if self.scanning or self.closing:
            return
        self.scanning = True
        dirs = self.prefs.model_dirs
        self._notice("Scanning local GGUF folders…")

        def discover() -> tuple[dict[str, Model], list[str], GPU | None]:
            models, errors = scan(dirs)
            self.store.save_models(models)
            return self.store.catalog(), errors, max(amd_gpus(), key=lambda gpu: gpu.free, default=None)

        def found(result: tuple[dict[str, Model], list[str], GPU | None]) -> None:
            self.scanning = False
            if self.closing:
                return
            if dirs != self.prefs.model_dirs:
                self.rescan(open_chat)
                return
            self.models, self.scan_errors, self.gpu = result
            self._choices()
            if open_chat is not None and self.chat is None:
                self._open(open_chat)
            elif self.chat and self.chat.model and self._selected() is None:
                self._open(self.chat.id)
            elif not any(not model.issue for model in self.models.values()):
                self._notice(
                    "No chat-ready GGUF models found. Add a folder or inspect the libraries scanned.",
                    "models",
                )
            elif self.scan_errors and any(Path(name).is_dir() for name in dirs):
                self._notice(
                    "Some model folders could not be scanned. Review the model library for details.", "models"
                )
            else:
                self._notice()

        def failed(exc: Exception) -> None:
            self.scanning = False
            self._error(exc, "settings")

        self.workers.submit(self.workers.disk, discover, found, failed)

    def model_library(self) -> None:
        dialog = QDialog(self)
        dialog.setWindowTitle("Local model library")
        dialog.resize(770, 500)
        layout = QVBoxLayout(dialog)
        layout.addWidget(label("Local model library", "title"))
        items = QListWidget()
        details = QPlainTextEdit()
        details.setReadOnly(True)
        items.currentItemChanged.connect(
            lambda item, _: details.setPlainText(
                item.data(USER_ROLE) if item else "\n".join(self.scan_errors)
            )
        )
        for model in sorted(self.models.values(), key=lambda item: (bool(item.issue), item.name.casefold())):
            item = QListWidgetItem(model.name + ("  ·  Unavailable" if model.issue else ""))
            lines = [
                model.path,
                f"{model.architecture or 'Unknown architecture'}  ·  {model.quant}",
                f"{model.parameters:,} parameters  ·  {model.size:,} bytes  ·  {model.context:,} native context",
                *model.shards[1:],
                model.issue,
            ]
            item.setData(USER_ROLE, "\n".join(line for line in lines if line))
            items.addItem(item)
        layout.addWidget(items, 2)
        layout.addWidget(details, 1)
        details.setPlainText(
            "\n".join(self.scan_errors) or "Select a model to see its metadata and location."
        )
        row = QHBoxLayout()
        row.addWidget(button("Model folders…", lambda: (dialog.accept(), self.preferences())))
        row.addWidget(button("Rescan", lambda: (dialog.accept(), self.rescan())))
        row.addStretch()
        row.addWidget(button("Close", dialog.accept))
        layout.addLayout(row)
        dialog.exec()

    def _server_menu(self) -> None:
        menu = QMenu(self)
        reload_action = menu.addAction("Reload model", self._load)
        reload_action.setEnabled(self._selected() is not None and self.operation is None)
        stop_action = menu.addAction("Unload model", self._unload)
        stop_action.setEnabled(self.info is not None or self.operation is not None)
        menu.addAction("Diagnostics…", self.diagnostics)
        menu.exec(self.reload.mapToGlobal(self.reload.rect().bottomLeft()))

    def diagnostics(self) -> None:
        dialog = QDialog(self)
        dialog.setWindowTitle("Inference diagnostics")
        dialog.resize(780, 430)
        layout = QVBoxLayout(dialog)
        layout.addWidget(label("Inference diagnostics", "title"))
        logs = QPlainTextEdit()
        logs.setReadOnly(True)
        logs.setPlainText(self.runtime.server.diagnostics() or "No inference startup log is available.")
        layout.addWidget(logs, 1)
        actions = QHBoxLayout()
        actions.addWidget(button("Stop server", lambda: (dialog.accept(), self._unload())))
        actions.addStretch()
        actions.addWidget(button("Close", dialog.accept))
        layout.addLayout(actions)
        dialog.exec()

    def _unload(self) -> None:
        self._invalidate()
        self.info, self.budget = None, None
        self._show_budget(None)
        self._stop_server()
        self._status("Stopped", "Model unloaded")

    def _health(self) -> None:
        if self.closing or not self.initialized or self.health_pending:
            return
        self.health_pending = True
        stamp = self.revision
        check_server = self.operation is None and self.info is not None

        def check() -> tuple[GPU | None, str]:
            gpu = max(amd_gpus(), key=lambda device: device.free, default=None)
            return gpu, self.runtime.server.health() if check_server else ""

        def checked(value: tuple[GPU | None, str]) -> None:
            self.health_pending = False
            self.gpu, error = value
            if error and stamp == self.revision and not self.closing:
                self.info = None
                self._show_budget(None)
                self._status("Error", "Inference server stopped")
                self._notice(error, "diagnostics")

        def failed(exc: Exception) -> None:
            self.health_pending = False
            LOG.warning("Health check failed error_type=%s", type(exc).__name__)

        pool = self.workers.inference if check_server else self.workers.disk
        self.workers.submit(pool, check, checked, failed)

    def closeEvent(self, event: QCloseEvent) -> None:
        if self.closed:
            event.accept()
            return
        event.ignore()
        if self.closing:
            return
        self.closing = True
        self.health_timer.stop()
        self._invalidate()
        self._status("Stopped", "Closing the inference server…")

        def done(_: Any) -> None:
            self.workers.inference.shutdown(wait=False, cancel_futures=True)
            self.workers.disk.shutdown(wait=False, cancel_futures=True)
            self.closed = True
            self.close()

        self.workers.submit(self.workers.inference, self.runtime.server.stop, done, done)
