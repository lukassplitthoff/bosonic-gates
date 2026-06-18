#!/usr/bin/env bash
# Build companion PDFs for all tutorial notebooks.
# Usage: bash build/build_all_pdfs.sh
# Set BINDER_FAST=1 to use reduced parameters for fast execution.

set -e

CONTENT_DIR="notebooks"
TEMPLATE=""  # add --template build/nbconvert_template.tplx when template is ready

echo "=== Building PDFs from tutorial notebooks ==="

for nb in "$CONTENT_DIR"/**/*.ipynb; do
    module_dir=$(dirname "$nb")
    pdf_dir="$module_dir/pdfs"
    mkdir -p "$pdf_dir"

    nb_name=$(basename "$nb" .ipynb)
    pdf_path="$pdf_dir/${nb_name}.pdf"

    echo "Converting: $nb -> $pdf_path"
    jupyter nbconvert --to pdf --execute \
        --ExecutePreprocessor.timeout=600 \
        --output "$nb_name" \
        --output-dir "$pdf_dir" \
        $TEMPLATE \
        "$nb" || echo "WARNING: Failed to convert $nb (skipping)"
done

echo "=== PDF build complete ==="
