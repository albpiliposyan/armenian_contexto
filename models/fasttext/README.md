# FastText Model

This directory is where the Armenian FastText model should live locally.

Expected files:

```text
cc.hy.300.bin
cc.hy.300.bin.gz
```

These files are intentionally ignored by Git because they are multi-GB binary
artifacts. Download them from the official FastText pretrained vectors page:

```text
https://fasttext.cc/docs/en/crawl-vectors.html
```

The engine expects the binary model at:

```text
models/fasttext/cc.hy.300.bin
```
