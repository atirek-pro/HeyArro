"""Log lines read ``[time][log_type]: message``."""

import logging
import re

from app.logging_setup import LOG_DATE_FORMAT, LOG_FORMAT


def _format(level, message):
    record = logging.LogRecord("app.test", level, __file__, 1, message, None, None)
    return logging.Formatter(LOG_FORMAT, LOG_DATE_FORMAT).format(record)


def test_a_line_is_time_then_type_then_the_message():
    line = _format(logging.INFO, "[Refinement] done in 12.41s")

    assert re.fullmatch(r"\[\d{2}:\d{2}:\d{2}\]\[INFO\]: \[Refinement\] done in 12\.41s", line)


def test_the_type_is_the_severity():
    assert "][ERROR]: boom" in _format(logging.ERROR, "boom")
    assert "][WARNING]: careful" in _format(logging.WARNING, "careful")
