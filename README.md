# change-detection-app

Before/after image change detection, with two interchangeable backends,
plus a Qwen VLM (via Ollama) description of what changed.

## Layout

```
change-detection-app/
├── main2.py            # classical CV approach: ECC alignment + SSIM diff
├── bit_cd_infer.py      # deep-learning approach: BIT_CD pretrained on LEVIR-CD
├── vendor/bit_cd/        # vendored BIT_CD model code + pretrained checkpoint
│   ├── models/           # networks.py, resnet.py, help_funcs.py (architecture only)
│   ├── checkpoints/BIT_LEVIR/best_ckpt.pt
│   └── NOTICE.md         # attribution / provenance for the vendored code
├── requirements.txt
├── .vscode/launch.json
└── .gitignore
```

`vendor/bit_cd` is self-contained — no separate `git clone` of the upstream
BIT_CD repo is required. See `vendor/bit_cd/NOTICE.md` for where it came from.

## Setup

```bash
pip install -r requirements.txt
ollama pull qwen3-vl:4b-instruct   # non-thinking tag — see note below
```

> **Which Qwen tag?** Use the non-thinking `qwen3-vl:4b-instruct` tag for the
> VLM explanation step. The plain `qwen3-vl:4b` tag has Ollama's `thinking`
> capability and reasons in a separate `thinking` channel before it answers.
> Recent Ollama builds ignore the API's `think: false` for that tag (on both
> `/api/generate` and `/api/chat`), so a modest token budget gets consumed
> entirely by reasoning and the model returns **no answer text** — which looks
> like "the VLM silently did nothing". If you only have `qwen3-vl:4b` pulled,
> either pull the `-instruct` tag above, or pass a much larger
> `--ollama-max-tokens` (the code auto-retries once at 4x the requested budget
> when it detects thinking-only output).

## Usage

```bash
# classical CV (SSIM) approach
python main2.py --before before.jpg --after after.jpg --out change_output

# BIT_CD deep-learning approach (LEVIR-CD building change detection)
python bit_cd_infer.py --before before.jpg --after after.jpg --out bit_cd_output
```

Both scripts share the same before/after/out/model/ollama-url CLI shape.
`bit_cd_infer.py` tiles large images into 256x256 patches (BIT_CD's native
training resolution) rather than resizing the whole image, since a single
resize badly degrades detection on full-scene images like LEVIR-CD's raw
1024x1024 releases.

`bit_cd_infer.py` is specifically a *building* change detector (LEVIR-CD is
a building change-detection benchmark) — it won't reliably flag vegetation,
road, or vehicle changes. `main2.py` is general-purpose but less semantically
aware (it flags any structural/visual difference, not just meaningful ones).
