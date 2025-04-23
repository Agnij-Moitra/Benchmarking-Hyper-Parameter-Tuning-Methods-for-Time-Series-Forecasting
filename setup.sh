#!/bin/bash
set -e
TARGET_DIR="./data/monash"
INDEX_FILE="$TARGET_DIR/index.txt"
if [ ! -f "$INDEX_FILE" ]; then
  echo "Error: index file not found at $INDEX_FILE"
  exit 1
fi
mkdir -p "$TARGET_DIR"
while IFS= read -r url; do
  if [ -z "$url" ]; then
    continue
  fi
  filename=$(basename "$url")
  filepath="$TARGET_DIR/$filename"
  if [ -f "$filepath" ]; then
    echo "File $filename already exists. Skipping download."
  else
    wget -O "$filepath" "$url"
  fi
  case "$filename" in
    *.zip)
      unzip -o "$filepath" -d "$TARGET_DIR"
      ;;
    *.tar.gz|*.tgz)
      tar -xzf "$filepath" -C "$TARGET_DIR"
      ;;
    *.tar)
      tar -xf "$filepath" -C "$TARGET_DIR"
      ;;
    *.gz)
      gunzip -kf "$filepath"
      ;;
    *.bz2)
      bunzip2 -k "$filepath"
      ;;
    *)
      echo "Unknown file type: $filename. Skipping extraction."
      ;;
  esac
done < "$INDEX_FILE"
find "$TARGET_DIR" -type f -name "*.zip" -exec rm -f {} \;
python init.py $(find ./data/monash/ -name "*.tsf")
find "$TARGET_DIR" -type f -name "*.tsf" -exec rm -f {} \;
autopep8 --in-place --aggressive --aggressive *.py
python featureExtraction.py
find "$TARGET_DIR" -type f -name "*.pkl" -exec rm -f {} \;