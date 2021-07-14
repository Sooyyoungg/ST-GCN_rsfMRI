import numpy as np

from keras.layers import *
from keras.models import *
from keras.optimizers import *
import keras.backend as K

from scipy import stats
from sklearn.model_selection import StratifiedKFold

if __name__ == "__main__":

    demo = np.loadtxt('demo_example.txt', delimiter='\t', dtype='str')

    # 기존: npy 파일 shape (roi=22개, timeseries=1200개)
    L = 383  # timeseries
    S = 0
    #data = np.zeros((demo.shape[0]*4, 1, L, 48, 1));  # (289, 1, 383, 48, 1)
    data = np.zeros((demo.shape[0]*4, 1, L, 333, 1));  # (289, 1, 383, 333, 1)
    label = np.zeros((demo.shape[0]*4,));

    # load all data
    idx = 0
    err = 0
    data_all = None

    for i in range(demo.shape[0]):
        for run in range(4):
            subject_string = demo[i, 0]
            # print(subject_string)

            # gordon (383,333)
            # harvardoxford (383,48)
            #filename_full = './abcd-fmriprep-rs-ts/' + subject_string + '/' + subject_string + '_ses-baselineYear1Arm1_task-rest_run-'+str(run+1)+'_space-MNIPediatricAsym_cohort-4_res-2_desc-preproc_atlas-harvardoxford_timeseries.npy'
            filename_full = './abcd-fmriprep-rs-ts/' + subject_string + '/' + subject_string + '_ses-baselineYear1Arm1_task-rest_run-'+str(run+1)+'_space-MNIPediatricAsym_cohort-4_res-2_desc-preproc_atlas-gordon_timeseries.npy'
            try:
                full_sequence = np.load(filename_full).T;  # (48, 383)

                if full_sequence.shape[1] < S + L:
                    continue
                    # 289개였다가 여기서 길이가 383이 안되는 55개 날아가고 234개 남음

                full_sequence = full_sequence[:, S:S + L];
                z_sequence = stats.zscore(full_sequence, axis=1)  # normalization -> z_sequence shape: (48, 383)

                if data_all is None:
                    data_all = z_sequence
                else:
                    data_all = np.concatenate((data_all, z_sequence), axis=1)

                data[idx, 0, :, :, 0] = np.transpose(z_sequence)  # subject, ? , timeseries, roi, ? (48, 383)
                label[idx] = demo[i, 1]
                idx = idx + 1
            except:
                print("no "+str(run+1)+"file in "+subject_string)

    # compute adj matrix
    # n_regions = 48 # harvardoxford atlas
    n_regions = 333  # gordon atlas
    A = np.zeros((n_regions, n_regions))
    for i in range(n_regions):
        for j in range(i, n_regions):
            if i == j:
                A[i][j] = 1
            else:
                A[i][j] = abs(np.corrcoef(data_all[i, :], data_all[j, :])[0][1])  # get value from corrcoef matrix
                A[j][i] = A[i][j]

    np.save('data/adj_matrix.npy', A)

    # split train/test and save data

    data = data[:idx]  # (234->234, 1, 383, 48, 1)
    label = label[:idx]
    print(data.shape)
    print(idx)

    skf = StratifiedKFold(n_splits=5, shuffle=True)
    fold = 1

    for train_idx, test_idx in skf.split(data, label):
        train_data = data[train_idx]
        train_label = label[train_idx]
        test_data = data[test_idx]
        test_label = label[test_idx]

        filename = 'data/gordon_train_data_all_run_' + str(fold) + '.npy'
        np.save(filename, train_data)
        filename = 'data/gordon_train_label_all_run_' + str(fold) + '.npy'
        np.save(filename, train_label)
        filename = 'data/gordon_test_data_all_run_' + str(fold) + '.npy'
        np.save(filename, test_data)
        filename = 'data/gordon_test_label_all_run_' + str(fold) + '.npy'
        np.save(filename, test_label)

        fold = fold + 1

