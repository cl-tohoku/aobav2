import argparse
import json
import logging
import pickle
import socket
import sys


logging.basicConfig(
    format="%(asctime)s #%(lineno)s %(levelname)s %(name)s :::  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


class DialogueServer():
    def __init__(self, port):
        self.contexts = []
        self.probs = []
        serversock = self.build_server(port)
        client_sock, client_address = serversock.accept()
        self.client_sock = client_sock
        logger.info("client was accepted!")

    def close(self):
        self.client_sock.close()

    def send_utter(self, send_data:dict):
        logger.info("\033[32m" + "send ... " + json.dumps(send_data, ensure_ascii=False) + "\033[0m")
        send_data = pickle.dumps(send_data)
        self.client_sock.sendall(send_data)

    def receive_utter(self) -> dict:
        received_data = self.client_sock.recv(1024)
        received_data = pickle.loads(received_data)
        logging.info("\033[34m" + "receive ... " + json.dumps(received_data, ensure_ascii=False) + "\033[0m")
        self.contexts.append(received_data["response"])
        self.probs.append(received_data["prob"])
        return received_data

    def build_server(self, port):
        host = socket.gethostbyname(socket.gethostname())
        logging.info("host: " + host)
        serversock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        serversock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        serversock.bind((host, port))
        serversock.listen(10)
        logging.info("Waiting for connections ...")
        return serversock


def main():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--turn", default=2, type=int, help="dialogue turn")
    parser.add_argument("--port", default=50000, type=int, help="port number")
    args = parser.parse_args()
 
    server = DialogueServer(args.port)

    # start utterance
    server_context = {"response":"server (user): こんにちは", "prob": 0}
    
    n_turn = 0
    while n_turn < args.turn:
        server.send_utter(server_context)
        client_response = server.receive_utter()
        server_context = {"response": "server (user): 東京タワーってどこにありますか？", "prob": 0}
        n_turn += 1

    server.close()


if __name__ == "__main__":
    main()
