import numpy as np

softmax_outputs = np.load('./softmax_outputs_cal.npy')
labels = np.load('./labels_cal.npy')
np.savez('./output/DeiT_model/DeiT_output_cal.npz', smx=softmax_outputs, labels=labels)