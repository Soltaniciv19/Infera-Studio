"""First-run setup, preferences, and conversation inference controls."""

from dataclasses import replace
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QMessageBox,
    QPlainTextEdit,
    QScrollArea,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
    QWizard,
    QWizardPage,
)

from infera_studio.catalog.gguf import Model
from infera_studio.config.preferences import Preferences
from infera_studio.config.settings import BACKENDS, CACHE_TYPES, CONTEXT_SIZES, InferenceSettings
from infera_studio.conversations.records import Chat, Message
from infera_studio.host.hardware import GIB, GPU
from infera_studio.inference.client import Budget
from infera_studio.inference.memory import MemoryEstimate, recommend
from infera_studio.inference.server import SAMPLERS, ServerInfo

from .widgets import button, label


def choice(items: tuple[str, ...], current: str) -> QComboBox:
    control = QComboBox()
    control.addItems(items)
    control.setCurrentText(current)
    return control


def make_note(text: str) -> QLabel:
    widget = label(text, "muted")
    widget.setWordWrap(True)
    return widget


class ExecutableForm(QWidget):
    def __init__(self, prefs: Preferences, discovered: dict[str, Path | None]):
        super().__init__()
        layout = QFormLayout(self)
        self.backend = choice(BACKENDS, prefs.backend)
        self.fields: dict[str, QLineEdit] = {}
        layout.addRow("Preferred backend", self.backend)
        for backend in BACKENDS:
            row = QWidget()
            row_layout = QHBoxLayout(row)
            row_layout.setContentsMargins(0, 0, 0, 0)
            entry = QLineEdit((prefs.binaries or {}).get(backend, ""))
            entry.setPlaceholderText(str(discovered.get(backend) or "Auto-detect or browse"))
            entry.setToolTip(str(discovered.get(backend) or "Choose an existing llama-server executable."))
            self.fields[backend] = entry
            row_layout.addWidget(entry, 1)
            row_layout.addWidget(button("Browse…", lambda _, target=entry: self._browse(target)))
            layout.addRow(backend, row)
        guidance = (
            "Leave a path blank to use automatic detection."
            if any(discovered.values())
            else "No llama-server executable was detected. Browse to an existing executable, "
            "or set one later in Settings."
        )
        layout.addRow(make_note(guidance))

    def _browse(self, field: QLineEdit) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Choose llama-server", field.text() or str(Path.home()))
        if path:
            field.setText(path)

    def values(self) -> dict[str, str]:
        return {
            name: str(Path(value).expanduser().absolute())
            for name, field in self.fields.items()
            if (value := field.text().strip())
        }


class DirectoryForm(QWidget):
    def __init__(self, dirs: tuple[str, ...]):
        super().__init__()
        layout = QVBoxLayout(self)
        layout.addWidget(make_note("Infera Studio scans these folders recursively for GGUF models."))
        self.paths = QListWidget()
        self.paths.addItems(dirs)
        self.paths.setMinimumHeight(140)
        layout.addWidget(self.paths, 1)
        actions = QHBoxLayout()
        actions.addWidget(button("Add folder…", self._add))
        actions.addWidget(button("Remove folder", self._remove))
        actions.addStretch()
        layout.addLayout(actions)

    def _add(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Choose a GGUF model folder", str(Path.home()))
        if path and path not in self.values():
            self.paths.addItem(path)

    def _remove(self) -> None:
        if self.paths.currentRow() >= 0:
            self.paths.takeItem(self.paths.currentRow())

    def values(self) -> tuple[str, ...]:
        return tuple(self.paths.item(i).text() for i in range(self.paths.count()))


class SetupWizard(QWizard):
    def __init__(self, parent: QWidget, prefs: Preferences, discovered: dict[str, Path | None]):
        super().__init__(parent)
        self.setWindowTitle("Set up Infera Studio")
        self.setWizardStyle(QWizard.WizardStyle.ModernStyle)
        self.resize(690, 420)
        self.prefs = prefs
        runtime = QWizardPage()
        runtime.setTitle("Connect llama.cpp")
        runtime.setSubTitle("Choose an existing llama-server installation for local inference.")
        self.executables = ExecutableForm(prefs, discovered)
        QVBoxLayout(runtime).addWidget(self.executables)
        self.addPage(runtime)
        files = QWizardPage()
        files.setTitle("Find your models")
        files.setSubTitle("Choose where Infera Studio should look for local model files.")
        self.directories = DirectoryForm(tuple(path for path in prefs.model_dirs if Path(path).is_dir()))
        QVBoxLayout(files).addWidget(self.directories)
        self.addPage(files)

    def values(self) -> Preferences:
        return replace(
            self.prefs,
            binaries=self.executables.values(),
            backend=self.executables.backend.currentText(),
            model_dirs=self.directories.values(),
            setup_complete=True,
        )


class PreferencesDialog(QDialog):
    def __init__(self, parent: QWidget, prefs: Preferences, discovered: dict[str, Path | None]):
        super().__init__(parent)
        self.prefs = prefs
        self.setWindowTitle("Infera Studio settings")
        self.resize(680, 430)
        layout = QVBoxLayout(self)
        tabs = QTabWidget()
        self.executables = ExecutableForm(prefs, discovered)
        self.directories = DirectoryForm(prefs.model_dirs)
        tabs.addTab(self.executables, "llama.cpp")
        tabs.addTab(self.directories, "Model folders")
        display = QWidget()
        form = QFormLayout(display)
        self.theme = choice(("Dark", "Light"), prefs.theme)
        self.reasoning = QCheckBox("Expand reasoning by default")
        self.reasoning.setChecked(prefs.show_reasoning)
        form.addRow("Appearance", self.theme)
        form.addRow(self.reasoning)
        tabs.addTab(display, "Appearance")
        layout.addWidget(tabs, 1)
        actions = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel
        )
        actions.accepted.connect(self.accept)
        actions.rejected.connect(self.reject)
        layout.addWidget(actions)

    def values(self) -> Preferences:
        return replace(
            self.prefs,
            model_dirs=self.directories.values(),
            binaries=self.executables.values(),
            backend=self.executables.backend.currentText(),
            theme=self.theme.currentText(),
            show_reasoning=self.reasoning.isChecked(),
            setup_complete=True,
        )


def gib(value: int | None) -> str:
    return "Unavailable" if value is None else f"{value / GIB:,.2f} GiB"


def memory_report(plan: MemoryEstimate | None, gpu: GPU | None, info: ServerInfo | None) -> str:
    lines = []
    if gpu:
        lines.extend(
            (
                f"GPU VRAM total   {gib(gpu.total)}",
                f"Currently used   {gib(gpu.used)}",
                f"Currently free   {gib(gpu.free)}",
            )
        )
    else:
        lines.append("AMDGPU VRAM counters are unavailable.")
    if plan:
        scope = "GPU" if plan.offload_fraction else "host"
        lines.extend(
            (
                "",
                f"Estimate at {plan.context:,} tokens ({scope})",
                f"Model weights   {gib(plan.weights)}",
                f"KV cache   {gib(plan.kv)}",
            )
        )
        if plan.recurrent:
            lines.append(f"Recurrent state   {gib(plan.recurrent)}")
        lines.extend(
            (f"Runtime margin   {gib(plan.runtime_margin)}", f"Estimated total   {gib(plan.required)}")
        )
        if plan.headroom is not None:
            lines.append(f"Estimated headroom before load   {gib(plan.headroom)}")
    if info:
        lines.extend(("", f"Successfully loaded slot   {info.context:,} tokens"))
        for device, kind, value in info.allocations:
            lines.append(f"{device} {kind} allocation   {value:,.1f} MiB")
        if info.observed_vram is not None:
            lines.append(f"Observed GPU usage change during load   {gib(info.observed_vram)}")
    return "\n".join(lines)


class ModelDialog(QDialog):
    def __init__(
        self,
        parent: QWidget,
        config: InferenceSettings,
        model: Model | None,
        info: ServerInfo | None,
        gpu: GPU | None = None,
    ):
        super().__init__(parent)
        self.config = config
        self.setWindowTitle("Model settings")
        self.resize(610, 590)
        layout = QVBoxLayout(self)
        if model:
            layout.addWidget(label(model.name, "title"))
            architecture = model.architecture or "unknown architecture"
            details = f"{architecture}   ·   {model.quant}   ·   {gib(model.size)}   ·   {len(model.shards)} GGUF file(s)"
            if model.parameters:
                details += f"   ·   {model.parameters / 1e9:.2f}B parameters"
            layout.addWidget(make_note(details))
            path = make_note(model.path)
            path.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            layout.addWidget(path)
        tabs = QTabWidget()
        launch = QWidget()
        launch_form = QFormLayout(launch)
        self.backend = choice(BACKENDS, config.backend)
        self.context = QComboBox()
        self.context.setEditable(True)
        self.context.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)
        context_sizes = {0, *CONTEXT_SIZES}
        if model and model.context:
            context_sizes.add(model.context)
            launch_form.addRow(make_note(f"GGUF advertises a native context of {model.context:,} tokens."))
        self.context.addItems(["Automatic" if size == 0 else str(size) for size in sorted(context_sizes)])
        self.context.setCurrentText(str(config.context) if config.context else "Automatic")
        self.output = QSpinBox()
        self.output.setRange(1, 2_147_483_647)
        self.output.setValue(config.output)
        self.layers = QSpinBox()
        self.layers.setRange(-1, 100_000)
        self.layers.setSpecialValueText("Automatic")
        self.layers.setValue(config.gpu_layers)
        self.flash = choice(("auto", "on", "off"), config.flash)
        self.cache_k = choice(CACHE_TYPES, config.cache_k)
        self.cache_v = choice(CACHE_TYPES, config.cache_v)
        for title, widget in (
            ("Backend", self.backend),
            ("Context tokens", self.context),
            ("Maximum output tokens", self.output),
            ("GPU layers", self.layers),
            ("Flash Attention", self.flash),
            ("Key cache", self.cache_k),
            ("Value cache", self.cache_v),
        ):
            launch_form.addRow(title, widget)
        if model:
            preview = make_note("")
            baseline = gpu
            if (
                gpu
                and info
                and info.plan.gpu
                and gpu.device == info.plan.gpu.device
                and info.observed_vram is not None
            ):
                baseline = replace(gpu, used=max(info.plan.gpu.used, gpu.used - max(0, info.observed_vram)))

            def show_estimate() -> None:
                try:
                    context = self.context.currentText().strip()
                    current = replace(
                        config,
                        context=0 if context == "Automatic" else int(context),
                        backend=self.backend.currentText(),
                        gpu_layers=self.layers.value(),
                        cache_k=self.cache_k.currentText(),
                        cache_v=self.cache_v.currentText(),
                    )
                except ValueError:
                    preview.setText("Enter a context size in tokens to see the estimate.")
                    return
                plan = recommend(model, current, baseline)
                scope = "GPU" if plan.offload_fraction else "host"
                detail = f"Estimated {scope} memory at {plan.context:,} tokens · {gib(plan.required)}"
                if plan.headroom is not None:
                    detail += f" · {gib(plan.headroom)} headroom"
                preview.setText(detail)

            for combo in (self.context, self.backend, self.cache_k, self.cache_v):
                combo.currentTextChanged.connect(show_estimate)
            self.layers.valueChanged.connect(show_estimate)
            show_estimate()
            launch_form.addRow(preview)
        launch_form.addRow(
            make_note(
                "Applying context or GPU settings reloads the model and checks the resulting slot size."
            )
        )
        tabs.addTab(launch, "Context and GPU")
        sampling = QWidget()
        sampling_form = QFormLayout(sampling)
        self.samplers: dict[str, tuple[QCheckBox, QSpinBox | QDoubleSpinBox | QLineEdit]] = {}
        supported = info.samplers if info else SAMPLERS - {"min_p"}
        for key, title, maximum in (
            ("temperature", "Temperature", 10.0),
            ("top_p", "Top-p", 1.0),
            ("top_k", "Top-k", 1_000_000),
            ("min_p", "Min-p", 1.0),
            ("repeat_penalty", "Repetition penalty", 10.0),
            ("seed", "Seed", 4_294_967_295),
        ):
            if key not in supported:
                continue
            current = getattr(config, key)
            enabled = QCheckBox("Override")
            enabled.setChecked(current is not None)
            if key == "seed":
                control = QLineEdit("" if current is None else str(current))
                control.setPlaceholderText("0–4294967295")
            elif isinstance(maximum, int):
                control = QSpinBox()
                control.setRange(0, maximum)
                control.setValue(current or 0)
            else:
                control = QDoubleSpinBox()
                control.setRange(0, maximum)
                control.setDecimals(3)
                control.setSingleStep(0.05)
                control.setValue(current or 0)
            control.setEnabled(enabled.isChecked())
            enabled.toggled.connect(control.setEnabled)
            row = QHBoxLayout()
            row.addWidget(enabled)
            row.addWidget(control, 1)
            sampling_form.addRow(title, row)
            self.samplers[key] = (enabled, control)
        self.thinking = choice(("Automatic", "On", "Off"), config.thinking)
        self.thinking.setToolTip("Automatic uses direct answers for models with a thinking switch.")
        self.thinking.setEnabled(
            bool((info and info.thinking_supported) or (model and model.has_thinking_switch))
        )
        if self.thinking.isEnabled():
            sampling_form.addRow("Thinking", self.thinking)
        sampling_form.addRow(
            make_note("Unset controls use GGUF sampling metadata when available, then llama.cpp defaults.")
        )
        tabs.addTab(sampling, "Sampling")
        self.system = QPlainTextEdit(config.system)
        self.system.setPlaceholderText("Optional system instructions. Leave blank for no system message.")
        self.system.setAccessibleName("System instructions")
        tabs.addTab(self.system, "System instructions")
        layout.addWidget(tabs, 1)
        actions = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Apply | QDialogButtonBox.StandardButton.Cancel
        )
        actions.button(QDialogButtonBox.StandardButton.Apply).clicked.connect(self.validate)
        actions.rejected.connect(self.reject)
        layout.addWidget(actions)

    def validate(self) -> None:
        try:
            context = self.context.currentText().strip()
            values = {}
            for key, (enabled, control) in self.samplers.items():
                values[key] = (
                    None
                    if not enabled.isChecked()
                    else (int(control.text()) if isinstance(control, QLineEdit) else control.value())
                )
            self.config = replace(
                self.config,
                backend=self.backend.currentText(),
                context=0 if context == "Automatic" else int(context),
                output=self.output.value(),
                gpu_layers=self.layers.value(),
                flash=self.flash.currentText(),
                cache_k=self.cache_k.currentText(),
                cache_v=self.cache_v.currentText(),
                thinking=self.thinking.currentText(),
                system=self.system.toPlainText(),
                **values,
            )
        except ValueError as exc:
            QMessageBox.warning(self, "Invalid settings", str(exc))
        else:
            self.accept()


class ContextDialog(QDialog):
    def __init__(
        self,
        parent: QWidget,
        chat: Chat,
        messages: list[Message],
        budget: Budget | None,
        plan: MemoryEstimate | None,
        gpu: GPU | None,
        info: ServerInfo | None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Context and memory")
        self.resize(580, 570)
        self.action = ""
        self.start = 0
        layout = QVBoxLayout(self)
        layout.addWidget(label("Active context", "title"))
        if budget:
            qualifier = "Estimated" if budget.estimated else "Tokenizer count"
            description = (
                f"Configured   {budget.context:,} tokens\n"
                f"Conversation and template   {budget.conversation:,}\n"
                f"System instructions   {budget.system:,}\n"
                f"Reserved for generation   {budget.reserved:,}\n"
                f"Remaining   {budget.remaining:,}\n{qualifier}"
            )
        else:
            description = "Context usage will appear after the model loads."
        counts = label(description)
        counts.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(counts)
        note = "Choose the first retained user turn. All messages remain in the conversation."
        if chat.summary:
            note += " An earlier-turn summary is active."
        layout.addWidget(make_note(note))
        self.cut = QComboBox()
        self.cut.addItem("Beginning of conversation", 0)
        for message in messages:
            if message.role == "user":
                first = " ".join(message.content.split())[:65]
                self.cut.addItem(first, message.id)
        self.cut.setCurrentIndex(max(0, self.cut.findData(chat.active_from)))
        layout.addWidget(self.cut)
        actions = QHBoxLayout()
        actions.addWidget(button("Apply context start", lambda: self._choose("cut")))
        self.summarize = button("Summarize earlier turns", lambda: self._choose("summarize"))
        self.summarize.setEnabled(
            info is not None and self.cut.currentData() > max(chat.active_from, self.cut.itemData(1) or 0)
        )
        self.cut.currentIndexChanged.connect(
            lambda: self.summarize.setEnabled(
                info is not None and self.cut.currentData() > max(chat.active_from, self.cut.itemData(1) or 0)
            )
        )
        actions.addWidget(self.summarize)
        layout.addLayout(actions)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        report = label(memory_report(plan, gpu, info), "muted")
        report.setWordWrap(True)
        report.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        scroll.setWidget(report)
        layout.addWidget(scroll, 1)
        bottom = QHBoxLayout()
        bottom.addWidget(button("Model settings", lambda: self._choose("settings")))
        bottom.addWidget(button("New chat", lambda: self._choose("new")))
        bottom.addStretch()
        bottom.addWidget(button("Close", self.reject))
        layout.addLayout(bottom)

    def _choose(self, action: str) -> None:
        self.action = action
        self.start = self.cut.currentData()
        self.accept()
