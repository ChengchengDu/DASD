import os
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import Dataset
import torch

PAD = "<pad>"
MASK = "<mask>"
CLS = "<s>"
SEP = "</s>"

def padding_to_max_length(tokens, max_length, pad_ids):
    """padding input_ids to max_length"""
    if len(tokens) <= max_length:
        diff_length = max_length - len(tokens)
        padding_ids = [pad_ids] * diff_length
        tokens.extend(padding_ids)     # 注意extend操作时不是需要赋值的
    else:
        tokens = tokens[: max_length]
    tokens = torch.tensor(tokens)

    return tokens


def trim_batch(
        input_ids,
        pad_token_id,
        attention_mask=None,
):
    """移除batch中行全为0的列"""
    keep_column_mask = input_ids.ne(pad_token_id).any(dim=0)   # dim=0表示的是处理所有的行
    if attention_mask is None:
        return input_ids[:, keep_column_mask]
    else:
        input_ids = input_ids[:, keep_column_mask]
        attention_mask = attention_mask[:, keep_column_mask]

        return (input_ids,attention_mask)


def encode_line(source_line, query_line, pad_id, max_source_length, tokenizer, pad_to_max_length=True):
    """Read original data, return input_ids which padding to max_length """
    # example_ids = []

    source_line = source_line.strip()
    query_line = query_line.strip()

    source_toks = tokenizer.tokenize(" " + source_line)
    query_toks = tokenizer.tokenize(" " + query_line)
    input_tokens = [CLS] + query_toks + [SEP] + [CLS] + source_toks + [SEP]

    if len(input_tokens) > max_source_length:
        input_tokens = input_tokens[:max_source_length]
    input_ids = tokenizer.convert_tokens_to_ids(input_tokens)

    if pad_to_max_length:
        input_ids = padding_to_max_length(input_ids, max_source_length, pad_id)
    else:
        input_ids = torch.tensor(input_ids)

    attention_mask = input_ids.ne(pad_id)

    return input_ids, attention_mask


def readfile(file_path):
    assert file_path is not None
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    return lines


class SummaryTestDataset(Dataset):
    """将train.source 和 train.target进行逐行处理得到input_ids和attention_mask
       只是单纯的seq2seq的摘要任务
    """
    def __init__(self,
                 tokenizer,
                 data_dir,
                 type_path="train",
                 max_source_length=1024,
                 max_target_length=15,
                 shuffle_query=False,
                 ):
        super(SummaryTestDataset, self).__init__()
        self.tokenizer = tokenizer
        self.pad_id = tokenizer.convert_tokens_to_ids([PAD])[0]
        source_path = os.path.join(data_dir, type_path + ".source")
        target_path = os.path.join(data_dir,  type_path + ".target")
        query_path = os.path.join(data_dir, type_path + ".query")
        self.source_lines = readfile(source_path)
        self.target_lines = readfile(target_path)
        self.query_lines = readfile(query_path)
        if shuffle_query:
            import random
            random.shuffle(self.query_lines)

        self.max_source_length = max_source_length
        self.max_target_length = max_target_length

    def __len__(self):
        return len(self.source_lines)

    def __getitem__(self, index):
        source_line = self.source_lines[index]
        query_line = self.query_lines[index]

        source_ids, source_mask = encode_line(
            source_line,
            query_line,
            self.pad_id,
            max_source_length=1024,
            tokenizer=self.tokenizer
        )
        input_dict = {
            "source_ids": source_ids,
            "source_mask": source_mask,
        }
        return input_dict

    @staticmethod
    def trim_seq2seq_batch(batch, pad_token_id):
        """去掉一个batch输入中某一列全部为pad的元素"""
        source_ids, source_mask = trim_batch(batch["source_ids"], pad_token_id, batch["source_mask"])

        return source_ids, source_mask

    def collate_fn(self, batch):
        """将batch中的tensor list进行stack 得到batch tensor可直接输入到模型中"""
        input_ids = torch.stack([x["source_ids"] for x in batch])
        masks = torch.stack([x["source_mask"] for x in batch])
        source_ids, source_mask = trim_batch(input_ids, self.pad_id, attention_mask=masks)
        input_dict = {
            "source_ids": source_ids,
            "source_mask": source_mask,
        }
        return input_dict


# 这一块重新修改一下 需要加入分布式的部分
# def get_dataloader(args, tokenizer, data_dir, type_path):
#
#     dataset = SummaryDataset(tokenizer=tokenizer,
#                              data_dir=data_dir,
#                              type_path=type_path,
#                             )
#     if type_path == "train":
#         sampler = RandomSampler(dataset) if args.local_rank == -1 else DistributedSampler(dataset)
#         data_loader = DataLoader(
#             dataset, sampler=sampler,
#             batch_size=args.train_batch_size,
#             collate_fn=dataset.collate_fn
#         )
#     else:
#         sampler = SequentialSampler(dataset)
#         data_loader = DataLoader(
#             dataset,
#             sampler=sampler,
#             batch_size=args.eval_batch_size,
#             collate_fn=dataset.collate_fn
#         )
#
#     return data_loader

