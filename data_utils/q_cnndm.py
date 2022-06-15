with open('/home/jazhan/code/journal_query_based_summarization/data_utils/dealed_data/cnndm/finetune_data/train.query', 'r', encoding='utf-8') as f:
    lines = f.readlines()

length = 0
all_ = len(lines)
for line in lines:
    length = length + len(line.strip().split(' '))
print(length / all_)

