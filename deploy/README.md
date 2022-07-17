# Telegram を用いた対話モデルの起動

- [対話システムライブコンペティション４#GettingStarted](https://dialog-system-live-competition.github.io/dslc4/gettingstart.html)

## 準備

```bash
# Mac のローカル環境で
$ brew cask install telegram
```

### bot 作成手順

1. Telegram の bot 検索画面で「BotFather」を検索
2. BotFather に対して `/newbot` とメッセージを送る
3. BotFather の返信メッセージに従って bot 名などを入力
4. Done! Congratulations on your new bot... というメッセージが来たら bot 作成完了
5. API トークンを控えておく（bot 実行時に使用）


## ディレクトリ構成

```yaml
# パラメータ引数
- configs/: 実行時のパラメータ引数（下記参照）
- outputs/: hydra のログ掃き溜め

# 実行関連
- run_telegram.py: telegram のボットを立ち上げる
- scripts/:
  - run_telegram.sh: run_telegram.py を実行するシェルスクリプト

# Python スクリプト
- telegrams/:
  - 
```


## 実行

```bash
$ python scripts/run_telegram.sh $API_TOKEN

# run_telegram.sh は、以下を実行する
$ python run_telegram.py \
  telegram.api_token ${API_TOKEN} \
  telegram.host ${HOST} \
  aobav1.port 42000 \
  dialogpt.port 45000 \
  fid.port 50000 \
  nttcs.port 40000
```

実行時のパラメータ引数は hydra を用いて `configs/` 下に管理されている。

```yaml
- configs/
  - deploy.yaml: dispatch
  - common/: 一般

  # モデルに関するパラメータ設定
  - aobav1/:
  - dialogpt/:
  - fid/:
  - nttcs/:
  - wiki_template/:

  # 後処理に関するパラメータ設定
  - post_processor/:
```
