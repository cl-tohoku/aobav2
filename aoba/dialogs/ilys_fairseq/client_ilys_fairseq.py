from os import path
import pickle
import socket
import sys

import logzero
from logzero import logger

from fairseq import options

from bot_our_fairseq_model import OurFairSeqModel


def main():
    parser = options.get_interactive_generation_parser()
    group = parser.add_argument_group("Dialogues")
    group.add_argument('--spm', type=path.abspath, default="/work02/SLUD2021/models/spm/8M_tweets.cr9999.bpe.32000.model", metavar="FP", help="Path to sentencepiece model")
    group = parser.add_argument_group("socket")
    group.add_argument('--port', default=42000)
    group.add_argument('--start_utter', default='こんにちは')
    group.add_argument('--host', default=socket.gethostbyname(socket.gethostname()))
    args = options.parse_args_and_arch(parser)

    fairseq_model = OurFairSeqModel(args)
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
            print('Fairseq model reset!')
            continue
        elif receive_data["response"] in ("/help", "/echo", "/log", "/reload", "/contexts"):
            continue

        print('Your Response: {}'.format(receive_data["response"]))
        context_list.append(receive_data["response"])
        prob_list.append(receive_data["prob"])

        responses = fairseq_model(context_list)
        print('My Response: {}'.format(responses))
        context_list.append(responses[0])

        data = {'response': responses, 'prob': 0}
        data = pickle.dumps(data)
        client.send(data)



if __name__ == "__main__":
    main() 
