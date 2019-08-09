import os
import torch


def train(epoch, model, optimizer, data, load_model_name, save_model_name, save_every):
    print('start')
    if load_model_name:
        model_name = os.path.join('model_storage', load_model_name)
        model.load_state_dict(torch.load(model_name))
    # 记录loss
    loss_sum = 0
    iter_num = 0
    for epoch in range(epoch):
        print(f'we are trainning epoch of {epoch+1}')
        for i, (sentence_in, targets) in enumerate(data):
            optimizer.zero_grad() # 也可以使用model.zero_grad
            loss = model.neg_log_likelihood(sentence_in, targets)
            loss_sum += loss
            iter_num += 1
            running_avg_loss = loss_sum/iter_num
            if (i+1) % 10 == 0:  # 每10个迭代输出当前loss
                print(f'>> Running average loss now is :{running_avg_loss.item()}')
            loss.backward()
            optimizer.step()
        if save_model_name:
            if (epoch+1) % save_every == 0:
                path_dir = os.path.join('model_storage', '{}'.format(save_model_name))
                torch.save(model.state_dict(), path_dir)




