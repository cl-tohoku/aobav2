#  Copyright (c) Microsoft Corporation. 
#  Licensed under the MIT license. 
import os

SP_TOKENS = ['<s>'] + [f'<ST{i}>' for i in range(1, 21)]

END_OF_TURN_TOKEN = '<|endofturn|>'
END_OF_TEXT_TOKEN = '<|endoftext|>'
PROJECT_FOLDER = os.path.dirname(__file__)
