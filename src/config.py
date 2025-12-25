class Config:
    num_classes = 10
    learning_rate = 0.001
    batch_size = 64
    num_epochs = 50
    ic_positions = [0, 1, 2, 3, 4]  # backbone layer indexleri
    ic_tau_start = 0.01
    ic_tau_end = 1.0
    pool_size = 4
