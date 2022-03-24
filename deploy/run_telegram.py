from os import path
import sys
import time
from typing import Dict

from logzero import logger

import telegram
from telegram import Update
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters, CallbackContext

sys.path.append(path.dirname(__file__))
from telegrams import create_args, UserContexts, DialogueAgent


def telegram_cmd(tag):
    def _telegram_cmd(func):
        def wrapper(*args, **kwargs):
            logger.info("\033[34m" + f"|--> {tag}" + "\033[0m")
            return func(*args, **kwargs)
        return wrapper
    return _telegram_cmd


class TelegramAgent(object):
    def __init__(
            self,
            api_token: str,
            bot_name: str = "あおば",
            dialogue_agent = None,
            max_len_contexts: int = 0,
            max_turn: int = 31,
    ):
        self.api_token: str = api_token
        self.bot_name: str = bot_name
        self.dialogue_agent = dialogue_agent
        self.max_len_contexts: int = max_len_contexts
        self.user_contexts_dict: Dict[int, UserContexts] = dict()
        self.max_turn = max_turn

    def create_user_contexts(self, update: Update, start: bool = False) -> UserContexts:
        user_id: int = update.message.from_user.id
        first_name: str = update.message.from_user.first_name
        last_name: str = update.message.from_user.last_name
        if start or user_id not in self.user_contexts_dict:
            return UserContexts(
                bot_name=self.bot_name,
                user_id=user_id,
                first_name=first_name,
                last_name=last_name,
                max_len_contexts=self.max_len_contexts
            )
        else:
            return self.user_contexts_dict[user_id]

    def update_user_contexts(self, user_contexts: UserContexts):
        user_id = user_contexts.user_id
        self.user_contexts_dict[user_id] = user_contexts

    @telegram_cmd("/start")
    def _start(self, update: Update, context: CallbackContext):
        user_contexts = self.create_user_contexts(update, start=True)
        response, user_contexts = self.dialogue_agent(utterance="", user_contexts=user_contexts)
        self.update_user_contexts(user_contexts)
        update.message.reply_text(response)

    @telegram_cmd("/help")
    def _help(self, update: Update):
        update.message.reply_text("""
            This bot was created for aobav2 (SLUD2021).
                - /help: This command
                - /start: Start a conversation with a bot.
                - /echo: Return the message as it was received.
                - /reload: Model was reloaded.
            If you add '--json_args' option and rewrite 'json_args' contents, the args contents will be changed.
                - /log: Output the dialogue logs to the file.
                - /contexts: Output the dialogue logs (contexts) to the telegram
        """)

    @telegram_cmd("/echo")
    def _echo(cls, update: Update, context: CallbackContext):
        received_text = update.message.text.replace("/echo", "", 1)
        received_text = received_text if received_text else "Please enter the text. (e.g. /echo hello!)"
        update.message.reply_text(received_text)

    @telegram_cmd("/reload")
    def _reload(self, update: Update, context: CallbackContext):
        self.dialogue_agent.reload()
        update.message.reply_text("The model was reloaded !")

    @telegram_cmd("/log")
    def _log(self, update: Update, context: CallbackContext):
        log_file = self.dialogue_agent.args.log_prefix + ".user_contexts"
        with open(log_file, "a") as fo:
            for user_id, user_context in self.user_contexts_dict.items():
                fo.write(f"--- Name = {user_context.user_name}, id = {user_id} ---" + "\n")
                fo.write(f"(0) /start" + "\n")
                for idx, (user, context) in enumerate(zip(user_context.full_users, user_context.full_contexts), 1):
                    fo.write(f"({idx}) {user}: {context}" + "\n")
                fo.write("\n")
        update.message.reply_text("Logging has been completed.")

    @telegram_cmd("/contexts")
    def _contexts(self, update: Update, context: CallbackContext):
        user_context = self.create_user_contexts(update)
        reply_texts = ["(0) /start"]
        for idx, (user, context) in enumerate(zip(user_context.full_users, user_context.full_contexts), 1):
            reply_texts.append(f"({idx}) {user}: {context}")
        update.message.reply_text("\n".join(reply_texts))

    @classmethod
    def unknown(cls, update: Update, context: CallbackContext):
        context.bot.send_message(
            chat_id=update.effective_chat.id,
            text="Sorry, I didn't understand that command."
        )

    def message(self, update: Update, context: CallbackContext):
        """ エージェント間でメッセージをやり取りする """
        received_text = update.message.text
        user_contexts = self.create_user_contexts(update)
        response, user_contexts = self.dialogue_agent(utterance=received_text, user_contexts=user_contexts)
        self.update_user_contexts(user_contexts)
        update.message.reply_text(response)

        # finish
        if user_contexts.n_turns == self.max_turn:
            unix_time = str(int(time.mktime(update.message["date"].timetuple())))
            user_id = str(update.message.from_user.id)
            user_name = update.message.bot.username
            update.message.reply_text(f"_FINISHED_:{unix_time}:{user_id}:{user_name}")
            update.message.reply_text("対話終了です．エクスポートした「messages.html」ファイルを，フォームからアップロードしてください．")

    @classmethod
    def error(cls, update: Update, context: CallbackContext):
        logger.warning('update: {}'.format(update))
        logger.warning("Caused error: {}".format(context.error))

    def run(self):
        updater = Updater(token=self.api_token, use_context=True)
        dispatcher = updater.dispatcher
        # CommandHandler を用いてコマンドに対する処理を登録
        dispatcher.add_handler(CommandHandler("start", self._start))
        dispatcher.add_handler(CommandHandler("help", self._help))
        dispatcher.add_handler(CommandHandler("echo", self._echo))
        dispatcher.add_handler(CommandHandler("reload", self._reload))
        dispatcher.add_handler(CommandHandler("log", self._log))
        dispatcher.add_handler(CommandHandler("contexts", self._contexts))
        # MessageHandler を用いてメッセージに対する処理を登録
        dispatcher.add_handler(MessageHandler(Filters.command, self.unknown))
        dispatcher.add_handler(MessageHandler(Filters.text, self.message))
        dispatcher.add_error_handler(self.error)
        # Bot 起動
        updater.start_polling()
        updater.idle()



def main():
    args = create_args()

    bot = telegram.Bot(token = args.api_token)
    bot_information = bot.get_me()
    bot_name = bot_information.first_name
    logger.info("Bot name: {}".format(bot_name))
    logger.info("Bot username: {}".format(bot_information.username))

    dialogue_agent = DialogueAgent(args, max_turn=args.max_turn)

    telegram_bot = TelegramAgent(
        api_token = args.api_token,
        bot_name = bot_name,
        dialogue_agent = dialogue_agent,
        max_len_contexts = args.max_len_contexts,
        max_turn = args.max_turn+1
    )
    telegram_bot.run()


if __name__ == "__main__":
    """ bash
    python $0 --api_token {API} --yaml_args {CFG}
    """
    main()
