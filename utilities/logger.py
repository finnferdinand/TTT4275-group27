#!/usr/bin/env python

__author__ = "Finn Ferdinand Sandvand and Christian Le"
__copyright__ = "Copyright 2023"
__license__ = "MIT"

class Logger(object):
    def __init__(self):
        self.filename = "output.txt"
        with open(self.filename, "w") as file:
            file.write("OUTPUT REPORT:\n==============\n\n") # clear file
    
    def write(self, string):
        with open(self.filename, "a") as file:
            file.write(string)
