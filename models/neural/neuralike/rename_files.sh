#!/bin/bash
mkdir -p all
for f in */*; do
    echo mv "$f" "${f%%/*}/${f%%/*}_${f##*/}"
    mv "$f" "all/${f%%/*}_${f##*/}"
done

