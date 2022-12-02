import time, threading
from cFunctions import GlobFunc


class ReadMessageConsumer(threading.Thread):
    def __init__(self):
        super().__init__()
        self.name = "Thread -- ReadMessageConsumer"
    def run(self):
        while True:
            print("reading...")
            GlobFunc.readMessage()
            time.sleep(2)
