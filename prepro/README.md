# データセットの前処理について

## Twitter 疑似対話データ
* https://io-lab.esa.io/posts/2193

```bash
python prepro_twitter.py \
    --ddir {ddir} \
    --basename {basename}
```


## 英語データ

* ダウンロード：`ParlAI/parlai/scripts/display_data.py` より
* パラレルコーパスの作成：
    ```bash
    $ cd {ROOT_REPOSITORY}
    $ cat datasets.yml
    $ for data in "blended_skill_talk" "convai2" "dailydialog" "empathetic_dialogues" "wizard_of_wikipedia"
        python prepro/create_parallel_corpus ${data} --dest {DEST}
    ```
* 翻訳：
* GIZA++：