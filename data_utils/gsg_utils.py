# 处理mask gsg
import nltk
import os
from random import randrange
import sys
import time
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
from tqdm import tqdm
import pandas as pd

from collections import defaultdict
from transformers import AutoTokenizer


from bert_score.utils import (
    get_model,
    get_tokenizer,
    get_idf_dict,
    bert_cos_score_idf,
    get_bert_embedding,
    lang2model,
    model2layers,
    get_hash,
    cache_scibert,
    sent_encode,
)

mask = "<mask>"

def read_file(path):
    with open(path, 'r', encoding="utf-8") as f:
        lines = f.readlines()
    return lines

def gsg(model, tokenizer, data_path, type="train", save_path=None):
    query_path = os.path.join(data_path, type + ".query")
    source_path = os.path.join(data_path, type + ".source")
    print("query_path", query_path)
    print("source_path", source_path)
    print("save_path", save_path)

    query_lines = read_file(query_path)
    source_lines = read_file(source_path)

    mask_source_lines = []
    remove_sents = []

    for query_line, source_line in tqdm(zip(query_lines, source_lines)):
        query_line = query_line.strip()
        source_line = source_line.strip()
        source_list = nltk.sent_tokenize(source_line)
        # if (not query_line) or (not source_line):
        #     continue
        if len(source_list) <= 1:
            index = randrange(0, len(source_list) + 1)
            remove_sents.append(' '.join(source_list) + "\n")
            source_list.insert(index, mask)
            mask_source_lines.append(' '.join(source_list) + "\n")
        else:
            source_list, remove_sent = remove_sent_by_bertscore(model, tokenizer, source_list, query_line)
            index = randrange(0, len(source_list) + 1)
            source_list.insert(index, mask)
            mask_source_lines.append(' '.join(source_list) + "\n")
            remove_sents.append(remove_sent + "\n")

    assert len(source_lines) == len(mask_source_lines)
    assert len(source_lines) == len(query_lines)

    with open(save_path + type + ".source", 'w', encoding='utf-8') as f:
        f.writelines(mask_source_lines)
    with open(save_path + type + ".query", 'w', encoding='utf-8') as f:
        f.writelines(query_lines)
    with open(save_path + type + ".starget", 'w', encoding='utf-8') as f:
        f.writelines(source_lines)
    with open(save_path + type + ".target", 'w', encoding='utf-8') as f:
        f.writelines(remove_sents)
        
def remove_sent_by_bertscore(model, tokenizer, source_list, query_line):
    
    max_score = 0
    max_index = 0
    for index, source_line in enumerate(source_list):
        score = get_bert_score(model, tokenizer, query_line, source_line)
        if score > max_score:
            max_score = score
            max_index = index
    remove_sent = source_list.pop(max_index)
   #  print("remove_sent", remove_sent)
    return source_list, remove_sent

def get_bert_score(model, tokenizer, query_line, source_line):
    p, r, f1 = score([source_line], [query_line], model, tokenizer)
    return f1[0]

def score(
    cands,
    refs,
    model, 
    tokenizer, 
    verbose=False,
    idf=False,
    device=None,
    batch_size=64,
    nthreads=4,
    all_layers=False,
    lang=None,
    return_hash=False,
    rescale_with_baseline=False,
    baseline_path=None,
    use_fast_tokenizer=False
):
    """
    BERTScore metric.
    Args:
        - :param: `cands` (list of str): candidate sentences
        - :param: `refs` (list of str or list of list of str): reference sentences
        - :param: `model_type` (str): bert specification, default using the suggested
                  model for the target langauge; has to specify at least one of
                  `model_type` or `lang`
        - :param: `num_layers` (int): the layer of representation to use.
                  default using the number of layer tuned on WMT16 correlation data
        - :param: `verbose` (bool): turn on intermediate status update
        - :param: `idf` (bool or dict): use idf weighting, can also be a precomputed idf_dict
        - :param: `device` (str): on which the contextual embedding model will be allocated on.
                  If this argument is None, the model lives on cuda:0 if cuda is available.
        - :param: `nthreads` (int): number of threads
        - :param: `batch_size` (int): bert score processing batch size
        - :param: `lang` (str): language of the sentences; has to specify
                  at least one of `model_type` or `lang`. `lang` needs to be
                  specified when `rescale_with_baseline` is True.
        - :param: `return_hash` (bool): return hash code of the setting
        - :param: `rescale_with_baseline` (bool): rescale bertscore with pre-computed baseline
        - :param: `baseline_path` (str): customized baseline file
        - :param: `use_fast_tokenizer` (bool): `use_fast` parameter passed to HF tokenizer
    Return:
        - :param: `(P, R, F)`: each is of shape (N); N = number of input
                  candidate reference pairs. if returning hashcode, the
                  output will be ((P, R, F), hashcode). If a candidate have
                  multiple references, the returned score of this candidate is
                  the *best* score among all references.
    """
    assert len(cands) == len(refs), "Different number of candidates and references"

    assert lang is not None or model_type is not None, "Either lang or model_type should be specified"

    ref_group_boundaries = None
    if not isinstance(refs[0], str):
        ref_group_boundaries = []
        ori_cands, ori_refs = cands, refs
        cands, refs = [], []
        count = 0
        for cand, ref_group in zip(ori_cands, ori_refs):
            cands += [cand] * len(ref_group)
            refs += ref_group
            ref_group_boundaries.append((count, count + len(ref_group)))
            count += len(ref_group)

    if rescale_with_baseline:
        assert lang is not None, "Need to specify Language when rescaling with baseline"


    if not idf:
        idf_dict = defaultdict(lambda: 1.0)
        # set idf for [SEP] and [CLS] to 0
        idf_dict[tokenizer.sep_token_id] = 0
        idf_dict[tokenizer.cls_token_id] = 0
    elif isinstance(idf, dict):
        if verbose:
            print("using predefined IDF dict...")
        idf_dict = idf
    else:
        if verbose:
            print("preparing IDF dict...")
        start = time.perf_counter()
        idf_dict = get_idf_dict(refs, tokenizer, nthreads=nthreads)
        if verbose:
            print("done in {:.2f} seconds".format(time.perf_counter() - start))

    if verbose:
        print("calculating scores...")
    start = time.perf_counter()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    all_preds = bert_cos_score_idf(
        model,
        refs,
        cands,
        tokenizer,
        idf_dict,
        verbose=verbose,
        device=device,
        batch_size=batch_size,
        all_layers=all_layers,
    ).cpu()

    if ref_group_boundaries is not None:
        max_preds = []
        for beg, end in ref_group_boundaries:
            max_preds.append(all_preds[beg:end].max(dim=0)[0])
        all_preds = torch.stack(max_preds, dim=0)

    use_custom_baseline = baseline_path is not None
    if rescale_with_baseline:
        if baseline_path is None:
            baseline_path = os.path.join(os.path.dirname(__file__), f"rescale_baseline/{lang}/{model_type}.tsv")
        if os.path.isfile(baseline_path):
            if not all_layers:
                baselines = torch.from_numpy(pd.read_csv(baseline_path).iloc[num_layers].to_numpy())[1:].float()
            else:
                baselines = torch.from_numpy(pd.read_csv(baseline_path).to_numpy())[:, 1:].unsqueeze(1).float()

            all_preds = (all_preds - baselines) / (1 - baselines)
        else:
            print(
                f"Warning: Baseline not Found for {model_type} on {lang} at {baseline_path}", file=sys.stderr,
            )

    out = all_preds[..., 0], all_preds[..., 1], all_preds[..., 2]  # P, R, F

    if verbose:
        time_diff = time.perf_counter() - start
        print(f"done in {time_diff:.2f} seconds, {len(refs) / time_diff:.2f} sentences/sec")

    if return_hash:
        return tuple(
            [
                out,
                get_hash(model_type, num_layers, idf, rescale_with_baseline,
                         use_custom_baseline=use_custom_baseline,
                         use_fast_tokenizer=use_fast_tokenizer),
            ]
        )

    return out

    
if __name__ == "__main__":
    ppath = "/home/jazhan/data/"
    data_name = "PrekshaNema25/nema/"
    save_path = os.path.join(ppath, "PrekshaNema25/new_gsg_data/")
    data_train_type = "train"
    model_type = "roberta-large"
    num_layers = 17    # base 10
    tokenizer = get_tokenizer(model_type)
    model = get_model(model_type, num_layers, all_layers=False)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    data_path = os.path.join(ppath, data_name)
    gsg(model, tokenizer, data_path=data_path, type=data_train_type, save_path=save_path)





