import scipy.io
import os

datasets = ['CVC-ClinicDB', 'CVC-ColonDB', 'ETIS-LaribPolypDB', 'Kvasir', 'CVC-300']
eval_results_path = 'EvaluateResults'

for dataset in datasets:
    print("-----------------------------------------------------------")
    print('Dataset: ', dataset)
    dataset_mat = dataset + '-mat'
    dataset_eval_path = os.path.join(eval_results_path, dataset_mat)
    list_file = [os.path.join(dataset_eval_path, f) for f in os.listdir(dataset_eval_path)]
    for file in list_file:
        mat = scipy.io.loadmat(file)
        model_name = file.split('/')[-1].split('.')[0]
        print(model_name, "-- meanDic: {:.3f}, meanIoU: {:.3f}, Sm: {:.3f}, maxEm: {:.3f}, MAE: {:.3f}".
        format(float(mat['meanDic']), float(mat['meanIoU']), float(mat['Sm']), float(mat['maxEm']), float(mat['mae'])))
        # print("wFm: {:.3f}".format(float(mat['wFm'])))
