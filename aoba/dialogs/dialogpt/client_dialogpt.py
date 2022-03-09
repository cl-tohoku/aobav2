'''
e.g. DialoGPT Model
'''
import argparse
from logzero import logger
from os import path
import pickle
import socket
import sys

sys.path.append(path.dirname(__file__))
from dialogpt4slud import DialoGptModel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', default=45000)
    parser.add_argument('--start_utter', default='こんにちは')
    parser.add_argument('--host', default=socket.gethostbyname(socket.gethostname()))
    parser = DialoGptModel.add_parser(parser)
    args = parser.parse_args()

    decoder = DialoGptModel(args)

    context_list, prob_list = [], []

    # socket
    host = args.host
    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    
    client.connect((host, args.port))
    print('connect server!')


    while True:
        receive_data = client.recv(4096)
        if not receive_data:
            continue

        # received response
        receive_data = pickle.loads(receive_data)

        if receive_data["response"] == "/start":
            context_list, prob_list = [], []
            print('DialoGPT model reset!')
            continue
        elif receive_data["response"] in ("/help", "/echo", "/log", "/reload", "/contexts"):
            continue

        # 第二話者の場合，最初の発話を対話履歴に追加しておく
        if receive_data["response"] != args.start_utter and not context_list:
            context_list.append(args.start_utter)
            prob_list.append(0.)

        print('Your Response: {}'.format(receive_data["response"]))
        context_list.append(receive_data["response"][-1])
        prob_list.append(receive_data["prob"])

        responses = decoder(context_list, num_beams=5)
        print('My Response: {}'.format(responses))
        context_list.append(responses[0])

        data = {'response': responses, 'prob': 0}
        data = pickle.dumps(data)
        client.send(data)


if __name__ == "__main__":
    main()
