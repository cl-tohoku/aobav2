# fairseq による seq2seq 学習

```bash
$ cd ilys_fairseq
$ conda create -n ilys python=3.8 -y
$ pyenv local {miniconda/*/ilys}
$ bash scripts/set_env.sh
```

## 実行方法

```yml
# qsub を使用する場合
bash scripts/pretrain_fulll_470M.sh qsub

# qrsh でデバッグする場合
bash scripts/pretrain_fulll_470M.sh
```

## 実行ファイル

```yml
- scripts/:
    - abci_setting.sh: abci での共有設定
    - set_env.sh: 環境構築用

    # pretrain
    - pretrain_full_2.7B.sh: 
    - pretrain_full_470M.sh: 

    # finetune
```