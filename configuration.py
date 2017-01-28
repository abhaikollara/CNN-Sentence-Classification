config = {
    'n_epochs' : 20,
    'kernel_sizes' : [3, 4, 5],
    'dropout_rate' : 0.5,
    'val_split' : 0.4,
    'edim' : 300,
    'n_words' : None,   #Leave as none
    'std_dev' : 0.05,
    'sentence_len' : 54,
    'n_filters'  : 100,
    'batch_size' : 50,
    'paths' : ['data/rt-polarity.pos', 'data/rt-polarity.neg'],
    'l2_lambda' : 3,
}