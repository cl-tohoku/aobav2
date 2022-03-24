# Telegram を用いた対話モデルの起動

- [対話システムライブコンペティション４#GettingStarted](https://dialog-system-live-competition.github.io/dslc4/gettingstart.html)

 準備

```bash
# Mac のローカル環境で
$ brew cask install telegram
```

## bot 作成手順

1. Telegram の bot 検索画面で「BotFather」を検索
2. BotFather に対して `/newbot` とメッセージを送る
3. BotFather の返信メッセージに従って bot 名などを入力
4. Done! Congratulations on your new bot... というメッセージが来たら bot 作成完了
5. API トークンを控えておく（bot 実行時に使用）

## 実行

```bash
API={telegram api}
HOST=$(hostname -I | awk -F ' ' '{print $1}')

python run_telegram.py --api_token ${API} --yaml_args ${configs.yml}

# ターミナル出力で以下が表示されるので、以下の手順を進める
# |--> Waiting for a connection from {client_name} model ::: {self.host}
```

`ROOT_REPOSITORY/aoba/dialogs` 下のプロジェクト `{project_path}` に移動して以下を実行

```bash
cd {project_path}
bash run_client.sh
```
