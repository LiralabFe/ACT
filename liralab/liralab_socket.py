import socket

class LiralabSocket:
    def __init__(self, port):
        self.port = port
        self.host = "127.0.0.1"
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

        self.socket.connect((self.host, self.port))

    def write(self, msg : str):
        if self.socket == None: return
        self.socket.sendall(msg.encode("utf-8"))

    def read(self):
        return self.socket.recv(1024).decode("utf-8")