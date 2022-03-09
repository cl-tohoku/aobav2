'''
e.g. Fid Model
'''
import argparse
from logzero import logger
from os import path
import pickle
import socket
import sys
from logzero import logger

from omegaconf import OmegaConf

sys.path.append(path.join(path.dirname(__file__), '../../'))
from fid4slud import FidModel
from knowledges import TopicExtractor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', default=50000)
    parser.add_argument('--start_utter', default='こんにちは')
    parser.add_argument('--host', default=socket.gethostbyname(socket.gethostname()))
    parser = FidModel.add_parser(parser)
    parser = TopicExtractor.add_parser(parser)
    args = parser.parse_args()

    retriever = TopicExtractor(args)

    decoder = FidModel(args)

    context_list, prob_list = [], []

    # socket
    host = args.host
    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    
    client.connect((host, args.port))
    print('connect server!')


    while True:
        try:
            receive_data = client.recv(4096)
            if not receive_data:
                continue

            # received response
            receive_data = pickle.loads(receive_data)
            
            if receive_data["response"] == "/start":
                context_list, prob_list = [], []
                print('FiD model reset!')
                continue
            if receive_data["response"] in ("/help", "/echo", "/log", "/reload", "/contexts"):
                continue

            # 第二話者の場合，最初の発話を対話履歴に追加しておく
            if receive_data["response"] != args.start_utter and not context_list:
                context_list.append(args.start_utter)
                prob_list.append(0.)

            print('Your Response: {}'.format(receive_data["response"]))
            context_list.append(receive_data["response"])
            prob_list.append(receive_data["prob"])

            relevant_passages = retriever.search(context_list[-1], top_k=10, only_dpr=True)
            input_data = FidModel.convert_retrieved_psgs(context_list[-1], relevant_passages)
            print('query: ', context_list[-1])
            print('relevant_passages: ', relevant_passages)
            response = decoder(input_data)
            print('My Response: {}'.format(response))
            context_list.append(response)

            data = {'response': response, 'prob': 0}
            data = pickle.dumps(data)
            client.send(data)
        except:
            continue


if __name__ == "__main__":
    main()
