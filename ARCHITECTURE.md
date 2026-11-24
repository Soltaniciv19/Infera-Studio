# Infera Studio architecture

The repository builds one locally installed application. Its Python backend is a modular monolith, with in-process calls between packages and a shared SQLite database. Electron uses its own renderer and main processes, and the backend supervises the local llama.cpp executable.

## Repository layout

| Path | Responsibility |
| --- | --- |
| `electron/` | Production desktop main process, sandboxed renderer, and preload API |
| `native/whisper/` | Bundled whisper.cpp worker for incremental dictation |
| `infera_studio/desktop/` | Python launcher and the Electron and optional Qt adapters |
| `infera_studio/application/` | Shared use cases for model loading, context budgets, generation, and summaries |
| `infera_studio/inference/` | llama.cpp process supervision, authenticated loopback HTTP, prompt construction, and inference telemetry |
| `infera_studio/catalog/` | Read-only GGUF inspection and discovery |
| `infera_studio/conversations/` | Shared conversation, message, attachment, and performance records |
| `infera_studio/config/` | Validated settings, durable preferences, and XDG paths |
| `infera_studio/host/` | Linux CPU, memory, and GPU measurements |
| `infera_studio/persistence/` | SQLite transactions, queries, and schema migrations |
| `packaging/`, `scripts/`, `tests/` | Debian metadata, build and verification commands, and regression tests |

Desktop adapters call the application layer and assemble the runtime's dependencies. The application coordinates inference and persistence. Inference works with conversation records and settings directly. SQLite also uses those records but owns their serialization and migrations. Keep shared use cases in `application/`, interface behavior in the desktop adapter, and Qt imports in `desktop/qt/`.

## Execution and storage

The Electron desktop uses a local HTML/CSS interface, a sandboxed renderer, and a small context-isolated preload API. The renderer draws the conversation, keeps unsent drafts in memory, and sends named requests to Electron's main process. Main validates the requesting frame and handles local file dialogs, clipboard access, and explicit link clicks.

Electron starts `infera_studio.desktop.electron.bridge` over private stdin/stdout pipes. `protocol.py` handles request IDs, result and event serialization, cancellation, and worker scheduling. `operations.py` dispatches supported requests, and `attachments.py` validates attachment snapshots. Separate workers handle inference, conversation lookups, and host telemetry. Cancellation can interrupt a running request even while the inference worker is busy.

Hugging Face requests have their own worker. `application/hugging_face.py` lists public GGUF variants, pins downloads to the selected revision, and verifies sizes, GGUF headers, and Hub checksums when available. It stages all parts together under the configured model folder and publishes them with a directory rename. The catalog skips staging directories and the renderer refreshes it after publication.

Dictation uses a separate Python worker thread in the same bridge process. The renderer's AudioWorklet sends 450 ms batches of microphone samples during recording and flushes the last partial batch on Stop. Python converts the samples to PCM, transcribes overlapping windows in a persistent whisper.cpp subprocess, and sends preview events over the bridge. The renderer appends the final transcript to the unsent draft once. Custom whisper-cli executables receive complete WAV snapshots for previews and the final transcript.

The application loads bundled assets through its local `infera-studio:` protocol. Chat Markdown disables raw HTML and remote images. External HTTP(S) links open in the user's browser on activation. The Debian package ships Electron and the production Python packages. The optional Qt interface is available from the source tree for regression comparison.

- `catalog/gguf.py` reads GGUF headers, metadata, and tensor dimensions without mapping weights. The adapter passes scan results to SQLite, which retains missing entries.
- `config/preferences.py` atomically replaces the XDG configuration file.
- `inference/server.py` manages one authenticated single-model llama.cpp server. Its guardian monitors the app's lifetime over a pipe and stops the owned process group on EOF.
- `inference/client.py` handles loopback HTTP, tokenization, streaming events, and cancellation. `inference/prompts.py` builds requests from selected turns and explicitly configured content.
- `inference/telemetry.py` normalizes timings, prompt progress, and cache reuse from the completion stream. The application combines them with local phase timings and stores the final snapshot with the assistant response.
- `persistence/sqlite.py` opens one SQLite connection per database operation, with foreign keys and a busy timeout. Add schema changes to `persistence/migrations.py`.

SQLite stores conversation settings and model paths, messages including separated reasoning, a context start message, and an optional requested summary. Editing or regenerating forks the conversation so the full source remains available. Model deletion is outside the scope of every database operation.

Electron's Agent mode is enabled by choosing a project folder in the native picker. The inference worker sends explicit function schemas and executes completed tool calls through `application/workspace.py`. Tool results re-enter the current completion and SQLite records the operation names, paths and outcomes. The worker anchors filesystem access to the project directory and publishes new files atomically. Ask mode and new branches start with no project access.

The Web search composer toggle lets the tool harness call `application/web_search.py` for the current request in either chat mode. The Python worker reads the Tavily key from local preferences and sends it in the HTTPS authorization header to Tavily. Preference responses to the renderer expose whether a key is configured. The harness returns bounded snippets and source URLs to the model and records the query and outcome in the response activity.

The app requests one llama.cpp slot, disables context shifting and uses the slot's reported context size. It counts a server-rendered prompt through the model tokenizer. When unavailable it labels its conservative local estimate. Over-budget requests stop before generation and preserve the draft and stored history.

The GPU estimate uses on-disk quantized weight bytes, cache types and attention dimensions. On Qwen3.5 it accounts for full attention layers and reserves recurrent state separately. Partial offload uses the layer fraction as an estimate, with a runtime margin. A successful load reports the observed context and any allocations available in the llama.cpp log. Unknown architectures or unavailable VRAM measurements start conservatively, with the runtime fitter where available.
