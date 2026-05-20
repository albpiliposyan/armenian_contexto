# FastText Model

This directory is where the Armenian FastText model should live locally.

Expected files:

```text
cc.hy.300.bin
cc.hy.300.bin.gz
```

These files are intentionally ignored by Git because they are multi-GB binary
artifacts. The recommended setup path is:

```bash
python scripts/run_project.py --download-model-only
```

The script asks before downloading. The compressed archive is about 4.2 GB and
the extracted binary is about 6.8 GB.

The engine expects the binary model at:

```text
models/fasttext/cc.hy.300.bin
```

Manual source: https://fasttext.cc/docs/en/crawl-vectors.html
