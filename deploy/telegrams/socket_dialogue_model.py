import socket 
import pickle 


class DialogueBotForSocket():
    def __init__(self, client_sock:socket.socket):
        self.contexts = []
        self.probs = []
        self.client_sock = client_sock

    @property
    def data_format(self):
        return {'response':str, 'prob':float}

    def send_utter(self, data):
        self.client_sock.sendall(data)

    def receive_utter(self) -> bytes:
        received_data = self.client_sock.recv(1024)
        data = pickle.loads(received_data)
        self.contexts.append(data["response"])
        self.probs.append(data["prob"])

        return received_data
