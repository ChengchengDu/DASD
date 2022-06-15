from asyncore import write
from cgi import print_environ
import sys
from bert_score import score


def read_file(path, block=5):
    with open(path, 'r', encoding="utf-8") as f:
        lines = f.readlines()
        new_lines = [line.strip() for line in lines]
    return new_lines
    
    new_lines = []
    if block > 1:
        for i in range(0, len(lines), block):
            new_lines.append(lines[i: i + block])
    else:
        new_lines = lines
    
    return new_lines

def write_file(path, content):
    with open(path, 'w', encoding="utf=8") as f:
        f.writelines(content)

def cal_bert_score(query_path, source_path, candi_target_path, block=5, data_index=100000):
    query_lines = read_file(query_path, block=1)
    source_lines = read_file(source_path, block=1)
    candi_target_lines = read_file(candi_target_path)
    print(len(candi_target_lines))
    print(len(query_lines))
    q_p, q_r, q_f1 = score(candi_target_lines, query_lines, lang='en', verbose=True)
    s_p, s_r, s_f1 = score(candi_target_lines, source_lines, lang='en', verbose=True)
    source_list = []
    query_list = []
    candi_targets_list = []
    for index, (qf, sf) in enumerate(zip(q_f1, s_f1)):
        if qf >= 0.80 and sf >=0.80:
            source_list.append(source_lines[index] + "\n")
            query_list.append(query_lines[index] + "\n")
            candi_targets_list.append(candi_target_lines[index] + "\n")
    # with open('/home/jazhan/data/PrekshaNema25/weak_super_data/bertscore_filter_weak_nema/val.source', 'w', encoding="utf-8") as f:
    #     f.writelines(source_list)
    # with open('/home/jazhan/data/PrekshaNema25/weak_super_data/bertscore_filter_weak_nema/val.query', 'w', encoding="utf-8") as f:
    #     f.writelines(query_list)
    # with open('/home/jazhan/data/PrekshaNema25/weak_super_data/bertscore_filter_weak_nema/val.target', 'w', encoding="utf-8") as f:
    #     f.writelines(candi_targets_list)
    print("filter length")
    print(len(source_list))
    source_write_path = '/home/jazhan/data/wikiref/filter_weak/' + str(data_index) + '.source'
    query_write_path = "/home/jazhan/data/wikiref/filter_weak/" + str(data_index) + ".query"
    candi_target_write_path = "/home/jazhan/data/wikiref/filter_weak/" + str(data_index) + ".target"
    

    # with open('/home/jazhan/data/wikiref/filter_weak/100000.source', 'w', encoding="utf-8") as f:
    #     f.writelines(source_list)
    # with open('/home/jazhan/data/wikiref/filter_weak/100000.query', 'w', encoding="utf-8") as f:
    #     f.writelines(query_list)
    # with open('/home/jazhan/data/wikiref/filter_weak/100000.target', 'w', encoding="utf-8") as f:
    #     f.writelines(candi_targets_list)

    with open(source_write_path, 'w', encoding="utf-8") as f:
        f.writelines(source_list)
    with open(query_write_path, 'w', encoding="utf-8") as f:
        f.writelines(query_list)
    with open(candi_target_write_path, 'w', encoding="utf-8") as f:
        f.writelines(candi_targets_list)
  

    print("query p r f1{},{},{}".format(q_p, q_r, q_f1))
    print("source p r f1{},{},{}".format(s_p, s_r, s_f1))





if __name__ == "__main__":
    # cal_bert_score(
    #     query_path="/home/jazhan/data/PrekshaNema25/weak_super_data/val.query",
    #     source_path="/home/jazhan/data/PrekshaNema25/weak_super_data/val.source",
    #     candi_target_path="/home/jazhan/data/PrekshaNema25/weak_super_data/val.target"
    # )
    data_index = sys.argv[1]
    print("data_index", data_index)
    cal_bert_score(
        query_path="/home/jazhan/data/wikiref/train_split/query/" + str(data_index) + ".query",
        source_path="/home/jazhan/data/wikiref/train_split/source/" + str(data_index) + ".source",
        candi_target_path="/home/jazhan/data/wikiref/train_split/pseudo_target/" + str(data_index) + "_pseudo.txt",
        data_index=data_index
    )

    # cal_bert_score(
    #     query_path="/home/jazhan/data/wikiref/train_split/query/100000.query",
    #     source_path="/home/jazhan/data/wikiref/train_split/source/100000.source",
    #     candi_target_path="/home/jazhan/data/wikiref/train_split/pseudo_target/100000_pseudo.txt"
    # )
    


    