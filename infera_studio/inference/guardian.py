"""Keep the inference process group tied to the lifetime of the desktop app."""

import os
import selectors
import signal
import subprocess
import sys


def stop(child: subprocess.Popen) -> None:
    try:
        os.killpg(child.pid, signal.SIGTERM)
    except ProcessLookupError:
        child.wait()
        return
    try:
        child.wait(timeout=4)
    except subprocess.TimeoutExpired:
        pass
    # The group may still contain descendants if its leader already exited.
    try:
        os.killpg(child.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    child.wait()


def main() -> int:
    secret_fd = int(sys.argv[1])
    try:
        child = subprocess.Popen(
            sys.argv[2:], stdin=subprocess.DEVNULL, start_new_session=True, pass_fds=(secret_fd,)
        )
    except OSError as exc:
        print(f"Could not start llama-server ({exc}).", file=sys.stderr, flush=True)
        return 127
    finally:
        os.close(secret_fd)
    try:
        with selectors.DefaultSelector() as selector:
            selector.register(sys.stdin.fileno(), selectors.EVENT_READ)
            while child.poll() is None:
                if selector.select(timeout=0.15):
                    break
    finally:
        stop(child)
    return child.returncode if child.returncode >= 0 else 128 - child.returncode


if __name__ == "__main__":
    raise SystemExit(main())
