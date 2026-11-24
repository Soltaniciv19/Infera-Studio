"""Serialized background work with queued Qt notifications."""

import logging
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any

from PySide6.QtCore import QObject, Signal

LOG = logging.getLogger(__name__)


class Workers(QObject):
    result = Signal(object, object, object)
    event = Signal(str, object)

    def __init__(self) -> None:
        super().__init__()
        self.disk = ThreadPoolExecutor(max_workers=1, thread_name_prefix="infera-studio-disk")
        self.inference = ThreadPoolExecutor(max_workers=1, thread_name_prefix="infera-studio-inference")
        self.result.connect(self._result)

    def submit(
        self,
        pool: ThreadPoolExecutor,
        job: Callable[[], Any],
        success: Callable[[Any], None],
        fail: Callable[[Exception], None],
    ) -> Future:
        def run() -> None:
            try:
                value = job()
            except Exception as exc:
                self.result.emit(fail, exc, True)
            else:
                self.result.emit(success, value, False)

        return pool.submit(run)

    def _result(self, callback: Callable, value: Any, failed: bool) -> None:
        if failed:
            LOG.info("Background operation failed error_type=%s", type(value).__name__)
        callback(value)
