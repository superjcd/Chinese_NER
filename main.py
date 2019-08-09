import torch
from argparse import ArgumentParser
import torch.optim as optim
from model import BiLSTM_CRF
from vocabulary import Vocabulary
from dataset import ResumeData
from train import train




if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--data', default='train', help='Choose data type to train your model', choices=['train', 'test', 'dev'])
    parser.add_argument('--epoch', default=10, help='Epochs to train your model')
    parser.add_argument('--load_model_name', type=str, help='If wanna load model stats before trainning')
    parser.add_argument('--save_model_name', type=str, help='Directory to save your model')
    parser.add_argument('--save_every', type=int, default=1,help='After n epoch to save you model, \
                                                   make sure you had type in save_model_name param first')
    args = parser.parse_args()
    print(f'we are gonna use the following arguments:\n{args.__dict__}\n')
    voc = Vocabulary.build_corpus(args.data)
    data = ResumeData(voc)
    model = BiLSTM_CRF(len(voc.word2id), voc.tag2id, 100, 100)
    optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=1e-4)
    train(args.epoch, model, optimizer, data, args.load_model_name, args.save_model_name, args.save_every)




