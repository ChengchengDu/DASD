import os
import random
import sys


def read_data(file_path):
    with open(file_path, 'r', encoding="utf-8") as f:
        lines = f.readlines()
    return lines

def write_data(data, write_dir, type=".source", type_data="train"):
    write_path = os.path.join(write_dir, type_data + type)
    with open(write_path, 'w', encoding="utf-8") as f:
        f.writelines(data)


def sample(data_path, sample_num, write_path, type_data="train"):
    print("type_data", type_data)
    source_path = os.path.join(data_path, type_data + ".source")
    query_path = os.path.join(data_path, type_data + ".query")
    target_path = os.path.join(data_path, type_data + ".target")
    if type_data == "train":
        itarget_path = os.path.join(data_path, type_data + ".itarget")

    source = read_data(source_path)
    query = read_data(query_path)
    target = read_data(target_path)
    if type_data == "train":
        itarget = read_data(itarget_path)
    length = len(source)
    sample_index = random.sample(range(0, length), sample_num)

    sample_source = [source[i] for i in sample_index]
    sample_query = [query[i] for i in sample_index]
    sample_target = [target[i] for i in sample_index]
    if type_data == "train":
        sample_itarget = [itarget[i] for i in sample_index]

    write_data(sample_source, write_dir=write_path, type=".source", type_data=type_data)
    write_data(sample_query, write_dir=write_path, type=".query", type_data=type_data)
    write_data(sample_target, write_dir=write_path, type=".target", type_data=type_data)
    if type_data == "train":
        write_data(sample_itarget, write_dir=write_path, type=".itarget", type_data=type_data)

if __name__ == "__main__":
    datasets = sys.argv[1]
    type_data = sys.argv[2]
    if datasets == "nema":
        data_path = "/home/jazhan/data/PrekshaNema25/nema/few_instance_nema/"
        write_path = "/home/jazhan/data/PrekshaNema25/nema/few_instance_nema/jnls_fewshot/"
    elif datasets == "wikiref":
        data_path = "/home/jazhan/data/wikiref/qfs_wiki/few_instance_wikiref/"
        write_path = "/home/jazhan/data/wikiref/qfs_wiki/few_instance_wikiref/jnls_fewshot/"
    for num in [10, 50, 100]:
        cur_write_path = os.path.join(write_path, str(num))
        if not os.path.exists(cur_write_path):
            os.makedirs(cur_write_path)
        sample(
            data_path=data_path,
            sample_num=num,
            write_path=cur_write_path,
            type_data=type_data,
        )



