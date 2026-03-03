#!/usr/bin/env bash
set -euo pipefail

uv run mc bench niah --adapter all --tasks s_niah_1,s_niah_2,s_niah_3 --context-lengths 4096,8192 --samples-per-length 4 --seed 0
uv run mc bench mqar --adapter all --samples 16 --num-pairs 8 --num-queries 2 --seed 0
uv run mc bench longbench --adapter all --tasks single_doc_qa,multi_doc_qa,code --samples-per-task 2 --seed 0
uv run mc bench retrieval --adapter all --datasets swde,squad,fda --truncation-lengths 512,1024 --samples-per-dataset 2 --seed 0
