from torch.utils.data import Dataset, dataloader


class ResumeData(Dataset):
    def __init__(self, word_lists, tag_lists):
        self.word_list = word_lists
        self.tag_lists = tag_lists

    def __getitem__(self, item):
        pass # TODO 返回word 及  tag 队列所对应的index序列

    def __len__(self):
        pass
