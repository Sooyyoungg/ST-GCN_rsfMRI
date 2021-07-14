import torch
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

A = np.load('./data/harvardoxford_test_label_1.npy')
print(A)