#!/usr/bin/env bash
python prott5_embedder.py --input sequences/some.fasta --output embeddings/residue_embeddings.h5
python prott5_embedder.py --input sequences/some.fasta --output embeddings/protein_embeddings.h5 --per_protein 1