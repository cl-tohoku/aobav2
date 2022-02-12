# GIZA++ を使用したアラインメント評価

## 設定

```bash
$ bash setup.sh
```

## 使い方
- NOTE: 日本語ファイルは予めトークン区切りにする必要がある

```bash
$ bash run_giza.sh en.context ja.context outputs
```

### 出力ファイル

#### \*.A3.final
This file contains sentence pairs and word alignments from the source language (English) into the target language (Japanese) using Viterbi Alignment, which is the most probable alignment (the one that maximizes the alignment probability).<br>

各行について
1. ソース文（en）およびターゲット文（ja）のトークン長と Viterbi Alignment によるスコア
2. ターゲット文（NOTE: ターゲットのトークンが少なくとも一つのソースに対応づけられる）
3. ソース文の各トークンにどのターゲットトークンが紐付けられているか

```
# Sentence pair (1) source length 20 target length 14 alignment score : 5.37486e-26
調子 は どう ? 調子 に 乗る ため に チーター を 追いかけ てる の 
NULL ({ 2 9 13 14 }) hi ({ }) , ({ }) how ({ 3 }) are ({ }) you ({ }) doing ({ }) ? ({ 4 }) i'm ({ }) getting ({ }) ready ({ }) to ({ }) do ({ }) some ({ }) cheetah ({ 5 6 7 8 10 }) chasing ({ 11 12 }) to ({ }) stay ({ }) in ({ }) shape ({ 1 }) . ({ }) 
```

#### その他の出力ファイル
- [README](https://github.com/moses-smt/giza-pp/blob/master/GIZA%2B%2B-v2/README) を参照

## 参考
- [README](https://github.com/moses-smt/giza-pp/blob/master/GIZA%2B%2B-v2/README)
- https://github.com/moses-smt/giza-pp
- https://sinaahmadi.github.io/posts/sentence-alignment-using-giza.html
- https://gist.github.com/mosmeh/80f239f7372e4caf87bf

