import os
from syslog import LOG_EMERG
from xml.etree.ElementPath import ops


def get_instance_labels(weak_path, query_path, save_path):
    weak_query_path = os.path.join(weak_path, "train.query")
    with open(weak_query_path, 'r', encoding="utf-8") as f:
        weak_query_lines = f.readlines()
    weak_target_path = os.path.join(weak_path, "train.target")
    with open(weak_target_path, 'r', encoding="utf-8") as f:
        weak_target_lines = f.readlines()
    
    with open(query_path, 'r', encoding="utf-8") as f:
        query_lines = f.readlines()
    
    instance_targets = []
    for query in query_lines:
        index = weak_query_lines.index(query)
        print(index)
        instance_target = weak_target_lines[index]
        instance_targets.append(instance_target)
    assert len(instance_targets) == len(query_lines)

    with open(save_path, 'w', encoding="utf-8") as f:
        f.writelines(instance_targets)
    

if __name__ == "__main__":
    weak_path="/home/jazhan/data/wikiref/filter_weak_all/"
    query_path = "/home/jazhan/data/wikiref/qfs_wiki/few_wikiref/jnls_fewshot/50/train.query"
    save_path = "/home/jazhan/data/wikiref/qfs_wiki/few_wikiref/jnls_fewshot/50/train.itarget"
    get_instance_labels(weak_path=weak_path, query_path=query_path, save_path=save_path)


