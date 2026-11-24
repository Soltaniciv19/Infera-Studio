# Infera Studio repository guidelines

- Keep Qt widgets in the desktop modules and host operations off the Qt thread.
- Keep GGUF discovery read-only. Treat model paths, metadata and rendered chat text as untrusted input.
- Preserve stored history when changing active context. Never add implicit system messages or tools.
- Keep server binding on loopback, use argument lists, and stop only the app's owned process group.
- Use explicit type annotations and parameterized SQLite statements. Add migrations for schema changes.
- Run `python3 -m pytest` and `scripts/build-deb` before packaging changes are complete.
- Keep generated artifacts under `dist/` and avoid checking in model weights or user state.
