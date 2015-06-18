
# adapted from http://stackoverflow.com/questions/616645/how-do-i-duplicate-sys-stdout-to-a-log-file-in-python

import sys

class Logger(object):
    def __init__(self, filename=None):
        self.filename = filename
        self.terminal = sys.stdout
        if self.filename is not None:
            self.log = open(filename, "a")

    def write(self, message):
        self.terminal.write(message)
        if self.filename != None:
            self.log.write(message)

sys.stdout = Logger()