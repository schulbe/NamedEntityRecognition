import logging
from datetime import datetime

class NerFormatter(logging.Formatter):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.base_string = "{passed_time} -- {message}"
        self.start_time = datetime.now()

    def format(self, record):
        record = record.__dict__
        vars = {
            'message': record.get('msg', None),
            'passed_time': str(datetime.fromtimestamp(record.get('created'))-self.start_time)
        }
        return self.base_string.format(**vars)