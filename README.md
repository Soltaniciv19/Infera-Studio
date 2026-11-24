<p align="center">
  <img src="packaging/infera-studio.png" alt="Infera Studio app icon" width="96" height="96">
</p>

# Infera Studio

Infera Studio is an Ubuntu desktop app for chatting with local GGUF models. It manages your `llama-server` process, streams responses using the model's own chat template, and stores conversations locally in SQLite. Chat works offline once you have a compatible model and `llama-server` executable.

## Install on Ubuntu 26.04 LTS (amd64)

If you already have the Debian package, run this from the directory containing it.

```sh
sudo apt install ./infera-studio_1.3.0_amd64.deb
```

To build and install from a GitHub checkout, use Node.js 22.12 or later, npm, and the `build-essential`, `cmake`, `curl`, and `dpkg-dev` packages. Run the following from the repository root. The first build fetches the pinned Whisper source and model into `dist/` and checks their SHA-256 hashes.

```sh
./scripts/build-deb
sudo apt install ./dist/infera-studio_1.3.0_amd64.deb
```

Close Infera Studio before upgrading an existing installation. The package bundles Electron and uses system Python 3.12 or later for the inference supervisor and storage. Existing Infera Studio conversations, settings, and model folders are reused after an upgrade.

## Get started

1. Open **Infera Studio** from the desktop application menu.
2. In the first-run settings, review the suggested `llama-server` executables and model folders. For a custom setup, select the executable for each backend you plan to use and add the folders containing your GGUF files.
3. Select a model below the message box and send a message.

When executable fields are blank, Infera Studio checks backend-specific executables on `PATH`, conventional per-user build folders, and a generic executable on `PATH`. GPU backends are verified against llama.cpp's device list. A GPU build may also be used for CPU inference.

## Chat and context

- Choose a model from the picker below the message box or open a conversation. Infera Studio starts one authenticated server on loopback and queues model switches so both models never allocate VRAM at the same time.
- Use **Enter** to send, **Shift+Enter** for a newline and **Stop** to cancel. Use a message's action to edit and resend or regenerate in a branch. The original conversation stays in the sidebar.
- Click the microphone beside **Send** to dictate. Words appear directly in the message draft as you speak. Click the microphone again to finish and edit the draft, or press **Esc** to discard the dictation. Each recording can run for up to five minutes.
- When a request exceeds the loaded context, increase the setting, choose the first retained turn, request a summary, or start a new chat. Context changes preserve stored history. Summary requests run locally, and the resulting summary is included in subsequent prompts only when explicitly applied.

### Offline voice input

The Debian installer includes [whisper.cpp](https://github.com/ggml-org/whisper.cpp) and the [multilingual Whisper base Q5_1 model](https://huggingface.co/ggerganov/whisper.cpp/blob/main/ggml-base-q5_1.bin). Dictation works offline from the first launch. Infera Studio transcribes on the CPU and keeps the resulting text in the draft until you send it.

**Settings → Voice input** shows which Whisper files are active and lets you select custom files. For dictation in a source checkout, run `./scripts/build-whisper` to prepare the bundled files before `npm start`.

### Attachments and tools

- Select **Ask** for regular chat. Click **Add files** below the message box to attach UTF-8 text such as source code, Markdown, CSV, and JSON. Send up to eight files, with a 256 KiB limit per file and a 512 KiB total. Infera Studio stores a snapshot with the user turn and counts its text against the active context.
- Select **Agent** and choose a project folder. Ask the model to find or read project files, or to create a project from scratch in an empty folder. Agent can create new text files and any needed subfolders. Existing files are protected from overwrites. Agent uses a tool-capable GGUF chat template and a compatible llama-server build. Each response shows its file activity. Click the folder name to change projects, or select **Ask** to revoke folder access for the conversation.
- Add your **Tavily API key** under **Settings → Web search**. With a tool-capable model, turn on **Web search** beside **Ask** and **Agent** for the next message. The model can send search queries to Tavily and use the returned snippets and source URLs in its reply. The conversation shows when a search ran.
- Use **Vision** beside the paperclip to attach PNG or JPEG images. Add the matching `mmproj` GGUF to a model folder, rescan, then select it from **Vision** or **Model settings**. Use a llama-server build that supports vision and image token counting. You can send up to four images, with a 4 MiB limit per image and an 8 MiB total. Infera Studio stores a local copy with the conversation.

## Models and performance

- In the model picker, use **Model library** to inspect metadata, inaccessible folders and unsupported files. **Rescan model folders** refreshes availability. A conversation whose model moved can use a replacement.
- Open **Settings → Hugging Face** to search public GGUF repositories by model name or repository ID. Select a quantization and a saved model folder. The app shows the download size, fetches all parts of a split variant, and refreshes the local model library after the download completes. You can cancel from the same tab.
- Use **Thinking** to toggle reasoning for the active conversation when its chat template supports the switch. Use **MTP** to toggle automatic multi-token prediction for models with a compatible draft head. Changing MTP reloads the model.
- Open **Model settings** to change the backend, set the context and output budget, tune GPU offloading and cache precision, or add system instructions. Unset sampling overrides use numeric GGUF hints when present and llama.cpp defaults otherwise. For templates with a thinking switch, **Automatic** requests a direct answer. **On** lets the model spend part of its output budget on reasoning.
- Infera Studio enables multi-token prediction (MTP) automatically when the GGUF has an embedded prediction head and the selected llama-server supports `draft-mtp`. If your model uses a separate head, add it to your model folders, rescan, and select the matching GGUF under **Model settings → MTP draft head**. The context readout shows MTP status. You can turn it off in Model settings.

### Monitoring and diagnostics

- Open the context readout to see the tokenizer budget, reserved output, memory estimates and last successful slot size. GPU memory counters are optional. The observed change during load can include other desktop activity. Automatic selects a context from metadata and the measured headroom, with a conservative fallback when dimensions are unknown. A successful load confirms allocation. It does not guarantee every future decode or manual configuration will fit.
- Click the monitor icon in the toolbar to open **System statistics**. The readout refreshes about every two seconds while the app is visible. It shows system RAM and CPU load and temperature, plus GPU load, VRAM and temperature from AMDGPU sysfs or NVIDIA's optional `nvidia-smi`. Choose readings under **Settings → Stats** and select a GPU when multiple devices report. Device readings include activity from other applications.
- Open **Request performance** under any measured response to inspect its saved timings, throughput and context usage. Use **Advanced stats** to follow the current request while it runs. A tilde marks an estimated token count, and a greater-than-or-equal sign indicates a lower bound while generation counts are unavailable.
- Open **Advanced stats** in the toolbar for line charts. **Overview** plots measured throughput and token counts over time and includes timing, context and KV cache details. **Recent requests** compares up to 12 responses in the conversation. Hover over a chart or focus it and use the arrow keys to inspect a measurement. Choose a response from the list or use **View request charts** under that response to see its details.
- The performance details show the KV buffer allocation reported at model load and the number of prompt tokens reused from the cache. **System statistics** also shows physical KV cell occupancy when the llama.cpp build exposes that gauge.
- Use the server status button to unload or reload a model and open the inference diagnostics. Startup failures also link to diagnostics.

## Local data

New profiles store configuration under `${XDG_CONFIG_HOME:-~/.config}/infera-studio`, conversations under `${XDG_DATA_HOME:-~/.local/share}/infera-studio`, and logs and Electron state under `${XDG_STATE_HOME:-~/.local/state}/infera-studio`. For each XDG location, the app reuses an existing `cortex-local` directory when the corresponding `infera-studio` directory is absent.

Conversation text is stored in the local database. Files use user-only permissions on initial creation. Package removal leaves these per-user files in place. To erase chats, use **Delete all conversations** from the conversation menu before uninstalling. To remove all application data, manually remove the directories used by your profile after quitting. Model discovery reads existing GGUFs. Hugging Face downloads write new GGUFs to the selected model folder.

## Develop and verify

Source development requires Node.js 22.12 or later, npm, Python 3.12 or later, and the Electron system libraries declared in the Debian package. Use `dpkg-dev` to build the package and optionally `desktop-file-utils` to validate its launcher.

Run the commands below from the `Infera Studio` checkout. `npm start` and `python3 -m infera_studio` both open the Electron desktop after dependencies are installed.

```sh
npm ci
npm start
python3 -m pytest
INFERA_TEST_ELECTRON=1 python3 -m pytest tests/test_electron.py
ruff check .
./scripts/build-deb
./scripts/check-deb
dpkg-deb --info dist/infera-studio_1.3.0_amd64.deb
desktop-file-validate packaging/infera-studio.desktop
```

The Electron integration test uses an isolated profile, a test GGUF header and a mock llama.cpp server. Source launches require Chromium sandbox support. On Ubuntu systems restricting unprivileged user namespaces, use the installed Debian launcher or a compatible system Chromium sandbox helper for source development. The Debian package includes its root-owned Chromium sandbox helper.

Run the optional real-server integration test with a small local chat GGUF or the model you plan to use. Each configured executable is tested sequentially.

```sh
export INFERA_TEST_MODEL="$HOME/AI/models/example.gguf"
export INFERA_TEST_VULKAN="$HOME/AI/lama-cpp/llama.cpp/build-vulkan/bin/llama-server"
export INFERA_TEST_HIP="$HOME/AI/lama-cpp/llama.cpp/build-hip/bin/llama-server"
# Optionally set INFERA_TEST_CPU to run the same test with CPU offload disabled.
python3 -m pytest -m integration -v
```

## Verify the installed package

1. Install on a clean Ubuntu 26.04 amd64 desktop, then open **Infera Studio** from the application menu. Check that the wizard can complete with a model folder and independently selected backend executables.
2. Load a model, click the microphone, and dictate a short phrase. Check that the words appear in the message draft while recording and can be edited when you stop.
3. Send a short request and check that tokens stream, the context count updates, and inference timings appear. Send a longer request and stop it. Close Infera Studio and verify that its llama-server exits.
4. Open Infera Studio again. Verify that the chat, model and settings persist. Rename it and find it by text search. Exercise **Model settings** with a second backend. If HIP fails with Flash Attention enabled, test the same configuration with it off.
5. Remove with `sudo apt remove infera-studio`. Verify that the application menu entry is gone. Reinstall and confirm the stored conversation reappears.

Infera Studio was developed on Ubuntu 26.04.1 with Python 3.14 and llama.cpp revision `94f3af3d0`. Automatic Flash Attention uses the disabled setting for HIP with Qwen3.5 on the detected Navi 21 device family after a reproducible decode abort in the tested build. An explicit setting takes precedence. Vulkan remains the default.

See [ARCHITECTURE.md](ARCHITECTURE.md) for the execution and persistence design, and the [llama.cpp server documentation](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md) for endpoint and build option details.
