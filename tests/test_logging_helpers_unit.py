"""Unit tests for the structured logging helpers."""

import logging

from histoweave.logging import configure_logging, get_logger, log_event


def test_configure_and_log_event(caplog) -> None:
    configure_logging(level="INFO", log_format="text")
    logger = get_logger("histoweave.test_logging_helpers")
    with caplog.at_level(logging.INFO):
        log_event(logger, logging.INFO, "unit.probe", "hello", value=1)
    assert any("unit.probe" in rec.message or "hello" in rec.message for rec in caplog.records)
