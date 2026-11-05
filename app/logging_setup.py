"""Log format: ``[time][log_type]: message``.

For example ``[14:52:13][INFO]: [Refinement] done in 12.41s: 51 segment(s) ...``.
The message keeps its own ``[Area]`` prefix (Input, Refinement, Selection), so a
line reads time, severity, then what produced it.
"""

import logging

LOG_FORMAT = "[%(asctime)s][%(levelname)s]: %(message)s"
LOG_DATE_FORMAT = "%H:%M:%S"


def configure_logging(level=logging.INFO):
    logging.basicConfig(level=level, format=LOG_FORMAT, datefmt=LOG_DATE_FORMAT)
