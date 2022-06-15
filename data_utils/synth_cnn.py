import json
import os
from tqdm import tqdm

def write_data(data, write_dir, data_type="train", source="source"):
    write_path = os.path.join(write_dir, data_type + "." + source)
    with open(write_path, 'w', encoding="utf-8") as f:
        f.writelines(data)

def synth_data(json_path, write_dir):
    with open(json_path, 'r', encoding="utf-8") as f:
        data = json.load(f)
    data = data["data"]
    articles = []
    queries = []
    targets = []

    for d in tqdm(data):
        paragraphs = d["paragraphs"]
        for para in paragraphs:
            context = para["context"] + "\n"
            # target = para["target"] # 原始摘要
            qas = para["qas"]
            for qs in qas:
                question = qs["question"] + "\n"
                summary = qs["summary"] + "\n"
                if question not in queries:
                    articles.append(context)
                    queries.append(question)
                    targets.append(summary)
   
    write_data(articles, write_dir, data_type="train", source="source")
    write_data(queries, write_dir, data_type="train", source="query")
    write_data(targets, write_dir, data_type="train", source="target")


if __name__ == "__main__":
    json_path = "/home/jazhan/code/journal_query_based_summarization/data_utils/dealed_data/cnndm/json/back_trans_json/cnndm-train-squad1.1.json"
    write_path = "/home/jazhan/code/journal_query_based_summarization/data_utils/dealed_data/cnndm/finetune_data/"
    synth_data(json_path=json_path, write_dir=write_path)










