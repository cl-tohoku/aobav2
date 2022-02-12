#!/bin/bash
   
set -eu
versions=(1 3 6 11 19)


for ver in ${versions[@]} ; do
    python aoba/knowledges/esearch/register_docs_async_icu_normalizer.py \
        --index_name "wikidump_pageview_threshold_${ver}" \
        --kb_file "/work02/SLUD2021/datasets/wikipedia/formated_wikidumps/formated_wikidump_threshold_${ver}.jsonl" \
        --config_file "aoba/knowledges/esearch/index_config_sudachi.json" \
        --host "localhost" \
        --port "9200"
    echo "${ver} is finished"
done