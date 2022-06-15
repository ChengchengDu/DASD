import os

from numpy import source


source_target = []
data_path ="/home/jazhan/data/PrekshaNema25/original_data/valid_summary"
with open(data_path, 'r', encoding="utf-8") as f:
    lines = f.readlines()
    for line in lines:
        line = ' '.join(line.split()[1:-1]).strip()
        source_target.append(line + "\n")

with open('/home/jazhan/data/PrekshaNema25/nema/val.target', 'w', encoding="utf-8") as wf:
    wf.writelines(source_target)

