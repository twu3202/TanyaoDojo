#!/bin/bash
# Assembles the Hugging Face Space in OUT from this repository:
#   bash deploy/hf_space/build_space.sh OUT WEIGHTS LIBRIICHI_SO MORTAL_MODEL_PY
# WEIGHTS: an inference copy of a Mortal-format checkpoint (mortal, current_dqn, config); LIBRIICHI_SO: libriichi
# built for Linux x86_64 / CPython 3.10; MORTAL_MODEL_PY: model.py of the same Mortal source tree.
# Nothing is uploaded here; publishing is a separate, deliberate step.
set -eu
OUT=$1; W=$2; SO=$3; MP=$4
HERE=$(cd "$(dirname "$0")" && pwd); REPO=$(cd "$HERE/../.." && pwd)
mkdir -p "$OUT"
cp "$HERE/app.py" "$HERE/tiles.py" "$HERE/requirements.txt" "$OUT/"
cp "$HERE/SPACE_README.md" "$OUT/README.md"
cp "$REPO/jax_rl/mjai_bot/decision_api.py" "$OUT/"
{ echo "# From Mortal (https://github.com/Equim-chan/Mortal), AGPL-3.0-or-later, Copyright (C) Equim."
  echo "# Local change in the riichi-eval build tree: obs version 5 is handled like version 4 (no effect on version-4 models)."
  cat "$MP"; } > "$OUT/model.py"
cp "$SO" "$OUT/libriichi.so"
cp "$W" "$OUT/model.pth"
printf '*.pth filter=lfs diff=lfs merge=lfs -text\n*.so filter=lfs diff=lfs merge=lfs -text\n' > "$OUT/.gitattributes"
( cd "$OUT" && md5sum model.pth libriichi.so 2>/dev/null || md5 model.pth libriichi.so )
ls -la "$OUT"
