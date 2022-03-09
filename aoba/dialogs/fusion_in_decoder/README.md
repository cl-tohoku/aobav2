# Fusion-in-Decoder (FiD)

- This directory is forked by [facebookresearch/FiD](https://github.com/facebookresearch/FiD)

<img src="https://files.esa.io/uploads/production/attachments/4896/2022/02/09/49957/27069d54-7b0a-4a89-8f73-16511d10dfa9.png" alt="fid" title="fusion-in-decoder">


```bash
$ cd aobav2/aoba/dialogs/fid

# pyenv local しておく
$ conda create -n FiD python=3.8 -y
$ pyenv local ${env}/FiD
$ bash scripts/set_env.sh
```

## データセット

### 取得
TBA

### 形式
以下のインスタンスからなる JSON/JSONL ファイルを使用

```json
{
    "id": "(str) 質問ID",
    "question": "(str) 質問",
    "target": "(str) answers から一つ選択した答え。ない場合はランダムに選択される。",
    "answers": "(List[str]) 答えのリスト",
    "ctxs": [{
        "title": "(str) Wikipedia 記事タイトル",
        "text": "(str) Wikipedia 記事",
        "score": "(float) retriever の検索スコア (ない場合は 1/idx で置換される。generator では使用されない。)"
    }]
}
```


## 学習

```bash
$ bash scripts/train_generator.sh configs/train_generator_slud.yml
```

## 評価

```bash
$ bash scripts/train_generator.sh configs/train_generator_slud.yml
```
