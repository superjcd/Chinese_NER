import torch
from torch.utils.data import Dataset
from vocabulary import Vocabulary

# this is what i want: ( word1, word2....), (tag1, tag2.... )

voc = Vocabulary.build_corpus('train')


class ResumeData(Dataset):
    def __init__(self, voc):
        self.voc = voc
        self.word_lists = voc.word_lists
        self.tag_lists = voc.tag_lists

    def __getitem__(self, item):
        # 获取每个字对应的id
        # TODO: 对UNK_TOKEN 进行处理
        _word_list = self.word_lists[item]
        word_list = [self.voc.word2id[word] for word in _word_list]
        # 获取每个tag对应的id
        _tag_list = self.tag_lists[item]
        tag_list = [self.voc.tag2id[tag] for tag in _tag_list]
        return torch.tensor(word_list, dtype=torch.long), torch.tensor(tag_list, dtype=torch.long)

    def __len__(self):
        return len(self.word_lists)





