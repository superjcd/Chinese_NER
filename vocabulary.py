import os
from setting import START_TAG, STOP_TAG
# word_list 和tag_list
class Vocabulary:
    def __init__(self, split, word_lists, tag_lists, word2id, tag2id):
        self.split = split
        self.word_lists = word_lists
        self.tag_lists = tag_lists
        self.word2id = word2id
        self.tag2id = tag2id

    @classmethod
    def build_corpus(cls, split, data_dir="./ResumeNER"):
        """读取数据"""
        if split not in ['train', 'dev', 'test']:
            raise ValueError("请从以下选项中选择['train', 'dev', 'test']")
        word_lists = []
        tag_lists = []
        with open(os.path.join(data_dir, split+".char.bmes"), 'r', encoding='utf-8') as f:
            word_list = []
            tag_list = []
            for line in f:
                if line != '\n':
                    word, tag = line.strip('\n').split()
                    word_list.append(word)
                    tag_list.append(tag)
                else:
                    assert len(word_list) == len(tag_list)
                    word_lists.append(word_list)
                    tag_lists.append(tag_list)
                    word_list = []
                    tag_list = []

        word2id = cls.build_map(word_lists)
        tag2id = cls.build_map(tag_lists)
        # 增加'START' 和 'STOP'
        tag2id[START_TAG] = len(tag2id)
        tag2id[STOP_TAG] = len(tag2id)
        return cls(split,word_lists, tag_lists, word2id, tag2id)

    @staticmethod
    def build_map(lists):
        maps = {}
        for list_ in lists:
            for e in list_:
                if e not in maps:
                    maps[e] = len(maps)
        return maps


if __name__ == '__main__':
    voc = Vocabulary.build_corpus('train')
    print(voc.tag2id)





