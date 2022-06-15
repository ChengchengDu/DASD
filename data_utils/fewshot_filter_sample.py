from asyncore import write
from cgi import print_environ
import sys
import os
from bert_score import score
from tqdm import tqdm


def read_file(path, block=1):
    with open(path, 'r', encoding="utf-8") as f:
        lines = f.readlines()
        new_lines = [line.strip() for line in lines]
    return new_lines

def write_file(path, content):
    with open(path, 'w', encoding="utf=8") as f:
        f.writelines(content)

def cal_bert_score(query_path, source_path, candi_target_path, weak_target_path, save_path, datasets="nema", block=5):
    print(1)
    query_lines = read_file(query_path, block=1)
    source_lines = read_file(source_path, block=1)
    candi_target_lines = read_file(candi_target_path)
    weak_target_lines = read_file(weak_target_path)
    # if datasets == "wikiref":
    query_lines = read_file(query_path, block=1)
    source_lines = read_file(source_path, block=1)
    candi_target_lines = read_file(candi_target_path)
    weak_target_lines = read_file(weak_target_path)
    print(len(candi_target_lines))
    print(len(query_lines))
    print(len(weak_target_lines))
    print("计算相似性")
    q_p, q_r, q_f1 = score(candi_target_lines, query_lines, lang='en', verbose=True)
    s_p, s_r, s_f1 = score(candi_target_lines, source_lines, lang='en', verbose=True)
    source_list = []
    query_list = []
    candi_targets_list = []
    weak_targets_list = []
    for index, (qf, sf) in enumerate(zip(q_f1, s_f1)):
        if qf >= 0.80 and sf >=0.80:
            source_list.append(source_lines[index] + "\n")
            query_list.append(query_lines[index] + "\n")
            candi_targets_list.append(candi_target_lines[index] + "\n")
            weak_targets_list.append(weak_target_lines[index] + "\n")

    print("filter length")
    print(len(source_list))
    source_write_path = os.path.join(save_path,"train.source")
    query_write_path = os.path.join(save_path,"train.query")
    target_write_path = os.path.join(save_path,"train.target")
    instance_target_write_path =  os.path.join(save_path,"train.itarget")

    with open(source_write_path, 'w', encoding="utf-8") as f:
        f.writelines(source_list)
    with open(query_write_path, 'w', encoding="utf-8") as f:
        f.writelines(query_list)
    with open(target_write_path, 'w', encoding="utf-8") as f:
        f.writelines(candi_targets_list)
    with open(instance_target_write_path, 'w', encoding="utf-8") as f:
        f.writelines(weak_targets_list)
  
    print("query p r f1{},{},{}".format(q_p, q_r, q_f1))
    print("source p r f1{},{},{}".format(s_p, s_r, s_f1))



if __name__ == "__main__":
    datasets = sys.argv[1]
    if datasets == "wikiref":
        query_path="/home/jazhan/data/wikiref/filter_weak_all/train.query"
        source_path="/home/jazhan/data/wikiref/filter_weak_all/train.source"
        candi_target_path="/home/jazhan/data/wikiref/filter_weak_all/train.gtarget"
        weak_target_path="/home/jazhan/data/wikiref/filter_weak_all/train.target"
        save_path="/home/jazhan/data/wikiref/qfs_wiki/few_instance_wikiref/"
    elif datasets == "nema":
        query_path="/home/jazhan/data/PrekshaNema25/weak_super_data/bertscore_filter_weak_nema_0.80/train.query"
        source_path="/home/jazhan/data/PrekshaNema25/weak_super_data/bertscore_filter_weak_nema_0.80/train.source"
        candi_target_path="/home/jazhan/data/PrekshaNema25/weak_super_data/bertscore_filter_weak_nema_0.80/train.gtarget"
        weak_target_path="/home/jazhan/data/PrekshaNema25/weak_super_data/bertscore_filter_weak_nema_0.80/train.target"
        save_path="/home/jazhan/data/PrekshaNema25/nema/few_instance_nema/"
    else:
        print("参数不对")
        

    cal_bert_score(
        query_path=query_path,
        source_path=source_path,
        candi_target_path=candi_target_path,
        weak_target_path=weak_target_path,
        save_path=save_path,
        datasets=datasets
    )


    