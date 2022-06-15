import os
from tqdm import tqdm
import sys

def get_instance_labels_to_gloden(weak_path, golden_path, save_path):
    weak_source_path = os.path.join(weak_path, "train.source")
    with open(weak_source_path, 'r', encoding="utf-8") as f:
        weak_source_lines = f.readlines()
    weak_target_path = os.path.join(weak_path, "train.target")
    with open(weak_target_path, 'r', encoding="utf-8") as f:
        weak_target_lines = f.readlines()
    
    source_path = os.path.join(golden_path, "train.source")
    with open(source_path, 'r', encoding="utf-8") as f:
        source_lines = f.readlines()
    target_path = os.path.join(golden_path, "train.target")
    with open(target_path, 'r', encoding="utf-8") as f:
        target_lines = f.readlines()
    
    targets = []
    for weak_source in tqdm(weak_source_lines):
        index = source_lines.index(weak_source)
        target = target_lines[index]
        targets.append(target)
    assert len(targets) == len(weak_source_lines)

    with open(save_path, 'w', encoding="utf-8") as f:
        f.writelines(targets)
    

if __name__ == "__main__":
    dataset = sys.argv[1]
    print(dataset)
    if dataset == "wikiref":
        weak_path="/home/jazhan/data/wikiref/filter_weak_all/"
        golden_path = "/home/jazhan/data/wikiref/qfs_wiki/train/"
        save_path = "/home/jazhan/data/wikiref/filter_weak_all/train.gtarget"
    elif dataset == "nema":
        weak_path="/home/jazhan/data/PrekshaNema25/weak_super_data/bertscore_filter_weak_nema_0.80/"
        golden_path = "/home/jazhan/data/PrekshaNema25/nema/"
        save_path = "/home/jazhan/data/PrekshaNema25/weak_super_data/bertscore_filter_weak_nema_0.80/train.gtarget"
    get_instance_labels_to_gloden(weak_path=weak_path, golden_path=golden_path, save_path=save_path)


    


    