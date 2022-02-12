from googletrans import Translator  # googletrans==4.0.0-rc1


class GoogleTranslator(object):
    def __init__(self):
        self.translator = Translator()

    def __call__(self, text:str):
        return self.translator.translate(text, dest='ja').text



if __name__ == '__main__':
    text = 'I am happy'
    translator = GoogleTranslator()
    translated_text = translator(text)
    
    print(text)
    print(translated_text)