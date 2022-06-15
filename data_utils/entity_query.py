#!/usr/bin/python
# -*- coding: UTF-8 -*-
import os
from tqdm import tqdm
import argparse
from nltk import data, tokenize
from nltk.featstruct import _is_sequence
from nltk.text import ContextIndex
from nltk.tree import Tree
from numpy.lib.utils import source
from requests.api import delete
from tqdm import tqdm

import jieba
import http.client
import hashlib
import urllib
import random
import json
import time
# from data_utils import write_samples

import os

import re
import html
from urllib import parse
import requests

import json

import random
from multiprocessing import Pool

import os
import hashlib

import gc

from nltk.tokenize import sent_tokenize

import argparse
import subprocess
import shutil


# 名词 名词复数  人称代词  名词代词
POS = ["NN", "NNS", "PRP", "NNP"]
NER_LABEL = [
    "PERSON", "ORGANIZATION", "LOCATION", "CITY", "PERCENT", "NUMBER", "MONEY", "DATE", "COUNTRY", "TIME",
    "STATE_OR_PROVINCE", "COUNTRY", "NATIONALITY"
]


def write_source_to_path(args):
    """将source target分别写到文件中"""
    source_path = os.path.join(args.data_dir, args.data_type + ".source")
    target_path = os.path.join(args.data_dir, args.data_type + ".target")
    save_path = args.save_path
    source_save_path = os.path.join(args.save_path, args.data_type + '_source')
    target_save_path = os.path.join(args.save_path, args.data_type + '_target')

    if not os.path.exists(source_save_path):
        os.makedirs(source_save_path)

    if not os.path.exists(target_save_path):
        os.makedirs(target_save_path)

    sources_new = []
    targets_new = []
    index_ = 0
    with open(source_path, 'r', encoding="utf-8") as source, \
            open(target_path, 'r', encoding="utf-8") as target:
        source_lines = source.readlines()
        target_lines = target.readlines()
        for index, (source_line, target_line) in tqdm(enumerate(zip(source_lines, target_lines))):
            if not source_line.strip() or not target_line.strip():
                continue
            if len(source_line.strip()) < len(target_line.strip()):
                continue
            # if len(source_line.strip().split()) <= 50:
            #     continue
            sources_new.append(source_line.strip() + "\n")
            targets_new.append(target_line.strip() + "\n")

            with open(os.path.join(source_save_path, str(index_) + ".txt"), 'w', encoding="utf-8") as sf:
                sf.write(source_line.strip())

            with open(os.path.join(target_save_path, str(index_) + ".txt"), 'w', encoding="utf-8") as tf:
                tf.write(target_line.strip())

            index_ += 1

    with open(os.path.join(save_path, args.data_type + ".source"), 'w', encoding="utf-8") as w:
        w.writelines(sources_new)

    with open(os.path.join(save_path, args.data_type + ".target"), 'w', encoding="utf-8") as w:
        w.writelines(targets_new)
    assert len(sources_new) == len(targets_new)
    assert len(sources_new) == index_
    print(f"after dealing, remain {index_} document")



def stanford_openie(params):
    """
    这个是针对一个目录下的多个文件进行处理
    主要是针对目标摘要抽取关系三元组
    """

    target_dir, ie_path, mapping_dir = params.target_dir, params.ie_dir, params.mapping_dir
    candidate_dir = os.path.abspath(target_dir)
    ie_dir = os.path.abspath(ie_path)
    if not os.path.exists(ie_dir):
        os.makedirs(ie_dir)
    print("Preparing to ie %s to %s..." % (candidate_dir, ie_dir))
    candidates = os.listdir(candidate_dir)
    # make IO list file
    print("Making list of files to ie...")
    map_file = os.path.join(mapping_dir, "mapping_for_corenlp_openie.txt")
    # map_file = "/home/jazhan/code/QaExsuBart/data/mapping_for_corenlp_openie.txt"
    with open(map_file, "w") as f:
        for c in candidates:  # 对每一个文件进行处理
            f.write("%s\n" % (os.path.join(candidate_dir, c)))  # 所有处理的文本文件的路径
    command = ['java', '-Xmx4g', '-cp', '/home/jazhan/stanford-corenlp-4.2.0/*',
               'edu.stanford.nlp.pipeline.StanfordCoreNLP', '-annotators', 'tokenize,ssplit,pos,lemma,depparse,ner,natlog,openie', 
               '-threads', '40',
               'openie.resolve_coref', 'true', 'openie.max_entailments_per_clause', '1', '-ner.useSUTime', 'false',
               'affinity_probability_cap', '0.8',
               '-ner.applyNumericClassifiers', 'false',
               '-filelist', map_file, '-outputFormat', 'json', '-outputDirectory', ie_dir]  # 关系抽取的过程
    print("IE %i files in %s and saving in %s..." % (len(candidates), candidate_dir, ie_dir))
    subprocess.call(command)
    print("Stanford CoreNLP IE has finished.")
    os.remove(map_file)

    # Check that the IE directory contains the same number of files as the original directory
    num_orig = len(os.listdir(candidate_dir))
    num_ie = len(os.listdir(ie_dir))
    if num_orig != num_ie:
        raise Exception(
            "The candidate directory %s contains %i files, but it should contain the same number as %s (which has %i files). Was there an error during tokenization?" % (
                candidate_dir, num_ie, candidates, num_orig))
    print("Successfully finished tokenizing %s to %s.\n" % (candidate_dir, ie_dir))

    gc.collect()


# 构建一个json的data
def format_to_entity_query(args):
    """
    从目标摘要中抽取实体或者名词短语作为query 
    """
    data_path = args.data_path
    corpus_type = args.corpus_type
    corpus_source_path = os.path.join(data_path, corpus_type + "_source")   # 小文件的路径  每一个原文和摘要对都写进了一个文件中
    corpus_target_path = os.path.join(data_path, corpus_type + "_target")
    target_ie_path = os.path.join(data_path, corpus_type + "_openie")      # 存放从target中抽取到的实体三元组的信息的文件夹

    source_files = os.listdir(corpus_source_path)
    target_files = os.listdir(corpus_target_path)
    target_ie_files = os.listdir(target_ie_path)

    # 对多有的路径文件进行排序  一一对应
    source_files.sort(key=lambda x: int(x.split('.')[0]))
    target_files.sort(key=lambda x: int(x.split('.')[0]))
    target_ie_files.sort(key=lambda x: int(x.split('.')[0]))

    source_files_list = [os.path.join(corpus_source_path, f)for f in source_files]
    target_files_list = [os.path.join(corpus_target_path, f) for f in target_files]
    target_ie_files_list = [os.path.join(target_ie_path, f) for f in target_ie_files]

    assert len(source_files_list) == len(target_files_list)
    assert len(source_files_list) == len(target_ie_files_list)

    source_lines = []
    entity_query_lines = []
    target_lines = []

    for source_file, target_file, ie_file in tqdm(zip(source_files_list, target_files_list, target_ie_files_list)):

        if not os.path.exists(ie_file) or not os.path.exists(source_file) or not os.path.exists(target_file):
            raise ValueError("ie_json file or source file not exits")

        source_id = int(source_file.strip().split('/')[-1].split('.')[0])
        target_id = int(target_file.strip().split('/')[-1].split('.')[0])
        ie_id = int(ie_file.strip().split('/')[-1].split('.')[0])
        assert (source_id == target_id) and (source_id == ie_id)

        with open(ie_file, 'r', encoding="utf-8") as ie_f, open(source_file) as sf, \
                open(target_file) as tf:
            # 获取ie信息
            ie_data = json.load(ie_f)
            # 获取原文档
            source_line = sf.readline().strip()
            target_line = tf.readline().strip()

            # 获取当前实体query
            entity_query = get_ie(ie_data)
            # 根据当前返回的字典构建json文件 data_item
            if entity_query == "none":
                continue
            else:
                source_lines.append(source_line + "\n")
                target_lines.append(target_line + "\n")
                entity_query_lines.append(entity_query + "\n")
    
    source_path = os.path.join(args.entity_save_path, corpus_type + ".source")
    query_path = os.path.join(args.entity_save_path, corpus_type + ".query")
    target_path = os.path.join(args.entity_save_path, corpus_type + ".target")

    with open(source_path, 'w', encoding='utf-8') as s:
        s.writelines(source_lines)
    
    with open(target_path, 'w', encoding='utf-8') as t:
        t.writelines(target_lines)

    with open(query_path, 'w', encoding='utf-8') as q:
        q.writelines(entity_query_lines)
    
    print('entity query has done!')




def get_query_summary_pair(source_list, ie_data, target_list):
    """
    source_list: 原文档分句之后的列表
    ie_data: 当前解析出的所有的三元组 实体等信息
    target_list: 目标摘要分句之后的列表
    对于解析出的所有的关系三元组 构建(query, document, answer, summary)pair
    """
    
    entity_query = get_ie(ie_data) 

    return entity_query



def get_ie(data):
    """一条data就是一个输入的信息 得到一条数据中**所有**的 ie中的token 实体 关系等信息"""

    # 得到当前摘要中的三元组所提及的所有实体信息 然后根据此选择三元组 同时保留
    # 实体三元组的选择: 1. 去掉代词  2. 去重  3. 长度选择
    nouns = []
    ners = []
    
    prons = ['he', 'his', 'she', 'her', 'has', 'it', 'its', 'cnn', 'i', 'my', 'your', 'some', 'they', 'them', 'we']

    sentences = data["sentences"]  # 包含7项的字典

    for sentence in sentences:
        cur_tokens = sentence["tokens"]
        for token in cur_tokens:
            tok = token["word"]
            if tok in prons:
                continue
            cur_pos = token["pos"].lower()
            if cur_pos.startswith('n') and token not in nouns:
                nouns.append(tok)
            ner = token['ner']
            if ner in NER_LABEL and token not in ners:
                ners.append(tok)

    if len(ners) == 0:
        return "none"
    else:
        return ' '.join(ners)







if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # write_source_to_path
    parser.add_argument(
        '--data_dir', default="/home/jazhan/code/journal_query_based_summarization/data_utils/dealed_data/cnndm/finetune_data/", type=str
    )
    parser.add_argument(
        '--save_path', default="/home/jazhan/data/synth-cnn-qfs/entity_query", type=str
    )
    parser.add_argument(
        '--data_type', default="val", type=str
    )
    # stanford_openie
    parser.add_argument(
        '--target_dir', default="/home/jazhan/data/synth-cnn-qfs/entity_query/val_target/", type=str
    )
    parser.add_argument(
        '--ie_dir', default="/home/jazhan/data/synth-cnn-qfs/entity_query/val_openie/", type=str
    )
    parser.add_argument(
        '--mapping_dir', default="/home/jazhan/data/synth-cnn-qfs/entity_query/", type=str
    )
    # format_to_entity_query
    parser.add_argument(
        '--data_path', default="/home/jazhan/data/synth-cnn-qfs/entity_query/", type=str
    )
    parser.add_argument(
        '--entity_save_path', default="/home/jazhan/data/synth-cnn-qfs/entity_query/ner_entity_data/", type=str
    )
    parser.add_argument(
        '--corpus_type', default="train", type=str
    )

    args = parser.parse_args()
    # write_source_to_path(args=args)
    # stanford_openie(args)
    format_to_entity_query(args)
