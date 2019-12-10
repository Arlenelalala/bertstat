import os
import torch
from Bertstat.bert.modeling import BertConfig, BertModel
import Bertstat.bert.tokenization as tokenization
from Bertstat.torchsummaryX.torchsummaryX import summary
from Bertstat.torchstat import stat
from tqdm import tqdm
import matplotlib.pyplot as plt


def get_bert(BERT_PT_PATH, do_lower_case=True):
    # if torch.cuda.is_available():
    #     device = "cuda"
    # else:
    device = "cpu"
    bert_config_file = os.path.join(BERT_PT_PATH, f'bert_config.json')
    bert_config = BertConfig.from_json_file(bert_config_file)
    model_bert = BertModel(bert_config)

    init_checkpoint = os.path.join(BERT_PT_PATH, f'pytorch_model.bin')
    state_dict = torch.load(init_checkpoint, map_location=device)

    vocab_file = os.path.join(BERT_PT_PATH, f'vocab.txt')
    tokenizer = tokenization.FullTokenizer(vocab_file=vocab_file, do_lower_case=do_lower_case)  # bert 词表

    # print(f"para name：\n{state_dict.keys()}")
    # print(f"{model_bert}")
    return state_dict, model_bert


def model_parm_nums(model):
    """
    Other methods of counting BERT parameter
    :param model: BERT model
    """
    def print_model_parm_nums_1(model):
        total = sum([param.nelement() for param in model.parameters()])
        print('Number of params: %s ' % total)
        # print('Number of params: %.2fM' % (total / 1e6))

    def print_model_parm_nums_2(L, H, A, d_1, vocab=30522, l_max=512):
        d_model = H
        d_k = H / A

        parameter_emb = (vocab + 4 + l_max) * d_model
        # print('parameter_emb:', parameter_emb)
        parameter_tra = (1 + d_model) * d_k * A * 4 + (5 + 2 * d_1) * d_model + d_1
        # print('parameter_tra:', parameter_tra)
        parameter_pooler = (1 + d_model) * d_model
        # print('parameter_pooler:', parameter_pooler)
        parameter = parameter_emb + L * parameter_tra + parameter_pooler
        print("parameter", parameter)

    def print_model_parm_nums_3(model):
        # size=(batch_size, seq_length)
        summary(model, torch.randint(low=0, high=30521, size=(1, 1), dtype=torch.long))

    # 1. By model.parameters()
    print_model_parm_nums_1(model)
    # 2. With formula
    print_model_parm_nums_2(L=12, H=768, A=12, d_1=3072)
    print_model_parm_nums_2(L=24, H=1024, A=16, d_1=4096)
    # 3.By torchsummaryX
    print_model_parm_nums_3(model)


def draw(state_dict=None, bert_path=None):
    if not state_dict and bert_path:
        state_dict, _ = get_bert(bert_path)
    save_path = 'save'
    if not os.path.isdir(save_path):
        os.mkdir(save_path)

    L = max([int(str(key).split('.')[2]) for key in state_dict.keys() if str(key).split('.')[0] == 'encoder']) + 1
    # layer = ['embeddings', 'pooler']

    print('Generating picture')
    for l in tqdm(range(L)):
        target = []
        for key, data in state_dict.items():
            if key.split('.')[2] == str(l):
                key = '.'.join(str(key).split('.')[-2:])
                try:
                    r, c = data.size()
                    data = data.reshape([1, r * c])
                    data = data.numpy().tolist()[0]
                except ValueError:
                    data = data.numpy().tolist()
                target.append((key, data))
        drawing_histogram(target, save_path, f'encoder.layer.{l}')
        drawing_box(target, save_path, f'encoder.layer.{l}')


def drawing_box(target, save_path, layer):
    all_data = [x[1] for x in target]
    fig, axes = plt.subplots(figsize=(20, 20))
    axes.boxplot(all_data, labels=[x[0] for x in target],
                 vert=False, patch_artist=False, whis=[5, 95],
                 showfliers=False, showmeans=True, meanline=True)
    # plt.show()
    plt.savefig(os.path.join(save_path, f'{layer}.box.png'))


def drawing_histogram(target, save_path, layer):
    fig, ax = plt.subplots(nrows=int(len(target)/4), ncols=4, figsize=(20, 20))
    fig.subplots_adjust(hspace=0.4, wspace=0.4)

    for i in range(4):
        for j in range(int(len(target)/4)):
            num = i * 4 + j
            key, data = target[num]
            ax[i, j].hist(data, bins=500, range=(0.2*min(data), 0.8*max(data)), color='b')
            ax[i, j].set_title(key)

    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f'{layer}.hist.png'))


def bert_stat(bert_path, input_size=(1, 1)):
    """
    :param bert_path:
    :param input_size: (batch_size, seq_length)
    """
    state_dict, model = get_bert(bert_path)
    stat(model, input_size)
    draw(state_dict)









