#!/usr/bin/env python

class Logger(object):
    def __init__(self):
        self.filename = "output.txt"
        with open(self.filename, "w") as file:
            file.write("OUTPUT REPORT:\n==============\n\n") # clear file
    
    def write(self, string):
        with open(self.filename, "a") as file:
            file.write(string)
