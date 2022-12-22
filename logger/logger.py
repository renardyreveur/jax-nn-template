import logging

# Logger filter
class Fmt_Filter(logging.Filter):
    def filter(self, record):
        record.levelname = '%s]' % record.levelname
        return True


# Blank line logging
def log_newline(self):
    root_logger = logging.getLogger()
    console_h = root_logger.handlers[-1]
    blank_h = logging.getLogger("blank").handlers[0]

    # Switch handler, output a blank line
    root_logger.removeHandler(console_h)
    root_logger.addHandler(blank_h)
    root_logger.info('')

    # Switch back
    root_logger.removeHandler(blank_h)
    root_logger.addHandler(console_h)
