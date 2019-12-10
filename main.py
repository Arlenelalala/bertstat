from Bertstat import bert_stat


if __name__ == '__main__':
    BERT_PT_PATH = ''
    # input_size: (batch_size, seq_length)
    bert_stat(BERT_PT_PATH, input_size=(1,1))