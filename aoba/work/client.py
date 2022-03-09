"""
クライアントを実装する上でのテンプレート
基本的にはクライアント側の対話システムの出力をdataに格納さえしていればOK
"""
import argparse
import json
import logging
import pickle
import socket
import sys


logging.basicConfig(
    format='%(asctime)s #%(lineno)s %(levelname)s %(name)s :::  %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO,
    stream=sys.stdout,
)

logger = logging.getLogger(__name__)


def create_client_response(server_context):
    return f"client (model): オウム：{server_context}"


def main():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--port", default=50000, type=int, help="port number")
    parser.add_argument("--start_utter", default="こんにちは", type=str, help="start utterance")
    args = parser.parse_args()

    context_list, prob_list = [], []

    host = socket.gethostbyname(socket.gethostname())
    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # connect server
    client.connect((host, args.port))
    logger.info("connect server!")

    while True:
        server_context = client.recv(4096)
        if not server_context:
            break

        # received response
        server_context = pickle.loads(server_context)

        # 第二話者の場合，最初の発話を対話履歴に追加しておく
        if server_context["response"] != args.start_utter:
            context_list.append(args.start_utter)
            prob_list.append(0.)

        logger.info("\033[32m" + "receive ... " + json.dumps(server_context, ensure_ascii=False) + "\033[0m")
        context_list.append(server_context["response"])
        prob_list.append(server_context["prob"])

        model_response = create_client_response(context_list[-1])
        send_data = {"response": model_response, "prob": 1.0}
        logger.info("\033[34m" + "send ... " + json.dumps(send_data, ensure_ascii=False) + "\033[0m")
        context_list.append(send_data["response"])
        prob_list.append(send_data["prob"])
        
        client.send(pickle.dumps(send_data))


if __name__ == "__main__":
    main()
