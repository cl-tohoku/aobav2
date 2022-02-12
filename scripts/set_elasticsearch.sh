#!/bin/bash

mkdir -p lib
cd lib

wget -nc https://artifacts.elastic.co/downloads/elasticsearch/elasticsearch-7.10.0-linux-x86_64.tar.gz \
  && tar -zxvf elasticsearch-7.10.0-linux-x86_64.tar.gz \
  && rm elasticsearch-7.10.0-linux-x86_64.tar.gz

cd elasticsearch-7.10.0
bin/elasticsearch-plugin install analysis-kuromoji
bin/elasticsearch-plugin install analysis-icu
bin/elasticsearch-plugin install https://github.com/WorksApplications/elasticsearch-sudachi/releases/download/v2.1.0/analysis-sudachi-7.10.0-2.1.0.zip
