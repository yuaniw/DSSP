#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../assets"

python _download.py

for archive in background_texture.zip embodiments.zip objects.zip; do
  unzip -n "$archive"
  rm "$archive"
done

cd ..
echo "Configuring asset paths ..."
python script/update_embodiment_config_path.py
