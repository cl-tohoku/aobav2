# ElasticSearch

`elasticsearch >= 7.10.0, <= 7.10.2`

## setup

```bash
cd $ROOT_REPOSITORY
bash scripts/set_elasticsearch.sh   # elasticsearch-7.10.0
```

### build server

```bash
$ROOT_REPOSITORY/lib/elalsticsearch-7.0.0/bin/elasticsearch
```

### register documents
wikitools で作成した jsonl 形式の知識ベースを elasticsearch に登録。
`kuromoji, icu_normalizer` によるトークナイズと正規化を行う。

```bash
python register_document_async_icu_normalizer.py \
  [--index_name] \
  [--kb_file] \
  [--delete_old_index (-d)] \               # 同名のインデックスが存在する場合に削除するか
  [--config_file=index_config.json]
  [--host=localhost] \
  [--port=9200]
```

## ドキュメントの検索
elasticsearch に登録されたドキュメントに対して検索を行う。

```bash
python es_search.py <index name> <query>
```

