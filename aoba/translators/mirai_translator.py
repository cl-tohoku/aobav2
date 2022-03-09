from mirai_translate import Client


class MiraiTranslator(object):
    def __init__(self, source='en', target='ja'):
        self.translator = Client()
        self.source_lang = source
        self.target_lang = target

    def __call__(self, text: str):
        return self.translator.translate(text, self.source_lang, self.target_lang)


if __name__ == '__main__':
    text = 'I am happy'
    translator = MiraiTranslator()
    translated_text = translator(text)

    print(text)
    print(translated_text)