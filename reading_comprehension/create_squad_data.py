# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""create squad train squad_data"""
import os
import json
#from src.luke.config import LukeConfig
from src.model_utils.config_args import args_config as args
from src.utils.utils import create_dir_not_exist
from transformers import RobertaTokenizer
from src.utils.entity_vocab import EntityVocab
from src.reading_comprehension.dataProcessing import build_data_change

args.wiki_link_db_file = "squad_data/enwiki_20160305.pkl"
args.model_redirects_file = "squad_data/enwiki_20181220_redirects.pkl"
args.link_redirects_file = "squad_data/enwiki_20160305_redirects.pkl"

args.data_dir = os.path.join(args.data, 'squad')

create_dir_not_exist(args.data)
create_dir_not_exist(args.data_dir)

args.bert_model_name = 'roberta-large'
args.max_mention_length = 30
with open('squad_data/metadata.json') as f:
    metadata = json.load(f)
entity_vocab = EntityVocab('squad_data/entity_vocab.tsv')
args.entity_vocab = entity_vocab
args.tokenizer = RobertaTokenizer('squad_data/vocab.json', 'squad_data/merges.txt')
build_data_change(args)
