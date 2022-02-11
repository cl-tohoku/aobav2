# SentencePiece

- https://github.com/google/sentencepiece
- https://github.com/google/sentencepiece/blob/master/doc/options.md


## Train SPM Model

```bash
$ python train_spm.py \
    --input <path_to_input_file> \
    --prefix <model_prefix> \
    --output_dir <path_to_destination> \
    --n_vocab 32000 \
    --coverage 0.9995 \
```

## Encode

```bash
$ MODEL=/work01/ryuto/slud2020/preprocess/spm/10M_tweets.cr9999.bpe.32000.model
$ echo "今日はいい天気ですね。" | python encode_spm.py --model $MODEL
今日は いい 天気 ですね 。

# with I/O file
$ python encode_spm.py --model $MODEL < $INPUT_FILE > $OUTPUT_FILE
```

## Decode

```bash
$ MODEL=/work01/ryuto/slud2020/preprocess/spm/10M_tweets.cr9999.bpe.32000.model
$ echo "今日はいい天気ですね。" | python encode_spm.py --model $MODEL | python detokenize_spm.py --model $MODEL
今日はいい天気ですね。
```
