# DialoGPT

This repository is forked from [microsoft/DialoGPT](https://github.com/microsoft/DialoGPT).

Zhang+'20 - DialoGPT: Large-Scale Generative Pre-training for Conversational Response Generation (ACL) [[ACL Anthology](https://aclanthology.org/2020.acl-demos.30/)][[arXiv](https://arxiv.org/abs/1911.00536)][[Microsoft Project](https://www.microsoft.com/en-us/research/project/large-scale-pretraining-for-response-generation/)][[ScrapBox](https://scrapbox.io/tohoku-nlp/Zhang+'20_DialoGPT:_Large-Scale_Generative_Pre-training_for_Conversational_Response_Generation_(ACL_2020))]


## Set up

```bash
conda create -n dialogpt python=3.8 -y
pyenv local {your pyenv}/dialogpt

# if abci
qrsh -g gcb50246 -l rt_C.small=1 -l h_rt=2:00:00
module load cuda/10.2/10.2.89 cudnn/7.6/7.6.5 gcc/7.4.0

bash scripts/set_env.sh
```

## データセット作成

DialoGPT を動かすためのデータベースを作成するために、まずは TSV ファイルを作成する。

```tsv
# head datasets/sample.tsv
0.0 <context>\t1.0 <response>
```

### tsv ファイルの作成

トークナイズ済みの `fi_context, fi_response` ファイルを準備する

```bash
bash scripts/merge_file.sh ${fi_context} ${fi_response} ${fo_merge}
```

### データベースの作成

作成した tsv ファイルを用いてデータベースを作成する

```bash
bash scripts/create_db.sh ${fi_tsv} ${max_length=128}
```

### End-to-End

Twitter 10K データからデータベースを作成する場合

```bash
# qsub を使用する場合
bash scripts/prepro_twitter_10K.sh qsub

# qsub を使用しない場合
bash scripts/prepro_twitter_10K.sh
```

## Train

```bash
bash scripts/train_twitter_10K.sh

# 以下のファイルが作成される
work/GPT2.1e-05.16.1gpu.2022-03-09214622/
  - tokenizer/
  - GPT2-pretrain-step-10.pkl
  - config.json
```

## Interact

```bash
DEST=work/GPT2.1e-05.16.1gpu.2022-03-09214622

python dialogpt4slud.py \
  --dgpt_tokenizer $DEST/tokenizer \
  --dgpt_model $DEST/GPT2-pretrain-step-10.pkl \
  --dgpt_config $DEST/config.json
```


## Models

- [`rinna/gpt2-japanese-medium`](https://github.com/rinnakk/japanese-pretrained-models) のモデルパラメータを DialoGPT の [`GPT2LMHeadModel`](https://huggingface.co/docs/transformers/model_doc/gpt2#transformers.GPT2LMHeadModel) にロードする
- 具体的には、`LSP_train.py` 実行時に `--ja` を指定。他、`gpt2_training/train_utils.py` の `load_rinna_medium` 関数を作成した。


### English

```python
>>> from functools import partial
>>> from demo_utils import download_model_folder
>>> download_model = partial(download_model_folder, DATA_FOLDER=MODEL_FOLDER)
>>> target_folder = download_model(model_size='small', dataset='multiref', from_scratch=False)
```

```bash
$ ls models/small
pytorch_model.bin config.json vocab.json small_ft.pkl merges.txt

$ cat models/small/config.json
{
  "initializer_range": 0.02,
  "layer_norm_epsilon": 1e-05,
  "n_ctx": 1024,
  "n_embd": 768,
  "n_head": 12,
  "n_layer": 12,
  "n_positions": 1024,
  "vocab_size": 50257
}
```

### Japanese
- [Question: Does it only work for English? #7](https://github.com/microsoft/DialoGPT/issues/7)


#### rinna/japanese-gpt2-medium
- [huggingface](https://huggingface.co/rinna/japanese-gpt2-medium)

```python
>>> from transformers import T5Tokenizer, AutoModelForCausalLM

>>> tokenizer = T5Tokenizer.from_pretrained("rinna/japanese-gpt2-medium")
>>> tokenizer.do_lower_case = True  # due to some bug of tokenizer config loading
>>> model = AutoModelForCausalLM.from_pretrained("rinna/japanese-gpt2-medium")

>>> model.config
GPT2Config {
  "_name_or_path": "rinna/japanese-gpt2-medium",
  "activation_function": "gelu_new",
  "architectures": [
    "GPT2LMHeadModel"
  ],
  "attn_pdrop": 0.1,
  "bos_token_id": 1,
  "embd_pdrop": 0.1,
  "eos_token_id": 2,
  "gradient_checkpointing": false,
  "initializer_range": 0.02,
  "layer_norm_epsilon": 1e-05,
  "model_type": "gpt2",
  "n_ctx": 1024,
  "n_embd": 1024,
  "n_head": 16,
  "n_inner": 4096,
  "n_layer": 24,
  "n_positions": 1024,
  "resid_pdrop": 0.1,
  "scale_attn_weights": true,
  "summary_activation": null,
  "summary_first_dropout": 0.1,
  "summary_proj_to_labels": true,
  "summary_type": "cls_index",
  "summary_use_proj": true,
  "task_specific_params": {
    "text-generation": {
      "do_sample": true,
      "max_length": 50
    }
  },
  "transformers_version": "4.9.2",
  "use_cache": true,
  "vocab_size": 32000
}
```

