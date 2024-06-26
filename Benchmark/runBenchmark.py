import ADP.d_experiment
from myutil import DataLoader

from MinMax import MinMax
from FFQS import FFQS
from ADP.d_experiment import experiment_adp
from COBRAS.experiment.COBRAS import COBRAS
from COBRA import cobra
from ADPE.a import ADPE

from joblib import Parallel, delayed

import os
import pandas as pd
import warnings
warnings.simplefilter('ignore')


def write_ari21(data, title, dir='ari21'):
    ari21 = {}
    ari21['dataset'].append(title)
    ari21['ari'].append(data['ari'][-1])
    ari21['interaction'].append(data['interaction'][-1])
    df = pd.DataFrame(ari21)
    df.to_csv(f'{dir}/{title}.csv')


def add_ari21(data, title, ari21):
    ari21['dataset'].append(title)
    ari21['ari'].append(data['ari'][-1])
    ari21['interaction'].append(data['interaction'][-1])
    return ari21


def run(file: str) -> None:
    title = file.split('.')[0]
    data, label, k = DataLoader.get_data_from_local(dataDir + '/' + file)
    data = data.values

    datalen = len(data)
    datalen = 1_000

    print(f'run on {file}')
    # print('run on MinMax')
    # MinMaxARI = MinMax.minmax(data, label, queries=MinMax.queries_cal(
    #     datalen), title=file.split('.')[0])
    # print('run on ADP')
    # write_ari21(MinMaxARI, f'{title}_MinMax')
    # print('run on ffqs')
    # ffqsARI = FFQS.ffqs(data, label, queries=FFQS.queries_cal(
    #     datalen), title=title)
    # write_ari21(ffqsARI, f'{title}_FFQS')
    print('run on ADP')
    alpha = 0.22
    l = 5
    theta = 0.00001
    ADPARI = experiment_adp.experiemnt_adp(
        data, label, alpha, l, theta, q=datalen)
    # os.makedirs('result/ADP', exist_ok=True)
    ADPPath = 'result/ADP'
    os.makedirs(ADPPath, exist_ok=True)
    pd.DataFrame(ADPARI).to_csv(
        f'{ADPPath}/{title}.csv', index=False)
    # ari21ADP = add_ari21(ADPARI, title, ari21ADP)

    print('run on ADPE')
    xi = [0.6, 0.8]
    alpha = [0.15, 0.30]
    s = 10
    l = 5
    theta = 0.00001
    ADPEARI = ADPE(data, label, xi, alpha, s, l, theta, q=datalen)
    ADPEPath = 'result/ADPE'
    os.makedirs(ADPEPath,
                exist_ok=True)
    pd.DataFrame(ADPEARI).to_csv(
        f'{ADPEPath}/{title}.csv', index=False)
    # ari21ADPE = add_ari21(ADPEARI, title, ari21ADPE)
    # write_ari21(ADPEARI, f'{title}_ADPE')

    print('run on cobras')
    COBRASARI = COBRAS(data, label, budget=datalen)
    # os.makedirs('result/COBRAS', exist_ok=True)
    COBRASPath = 'result/COBRAS'
    os.makedirs(COBRASPath, exist_ok=True)
    pd.DataFrame(COBRASARI).to_csv(
        f'{COBRASPath}/{title}.csv', index=False)
    # ari21COBRAS = add_ari21(COBRASARI, title, ari21COBRAS)


if __name__ == "__main__":
    dataDir = 'G:/data/datasets/UCI/middle/data'
    # data = 'iris.csv'

    # for file in [data]:
    ari21ADP = {'dataset': [], 'ari': [], 'interaction': []}
    ari21ADPE = {'dataset': [], 'ari': [], 'interaction': []}
    ari21COBRAS = {'dataset': [], 'ari': [], 'interaction': []}

    files = os.listdir(dataDir)
    files = ['Segmentation.csv', 'Waveform-5000-C3.csv', 'OptDigits.csv',
             'EEG Eye State.csv', 'Avila.csv', 'Letter Recognition.csv']

    Parallel(n_jobs=3, batch_size=2)(delayed(run)(file) for file in files)

    # for file in os.listdir(dataDir):
    #     title = file.split('.')[0]
    #     data, label, k = DataLoader.get_data_from_local(dataDir + '/' + file)
    #     data = data.values

    #     datalen = len(data)
    #     datalen = 1_000

    #     if datalen > 1e3:
    #         continue

    #     print(f'run on {file}')
    #     # print('run on MinMax')
    #     # MinMaxARI = MinMax.minmax(data, label, queries=MinMax.queries_cal(
    #     #     datalen), title=file.split('.')[0])
    #     # print('run on ADP')
    #     # write_ari21(MinMaxARI, f'{title}_MinMax')
    #     # print('run on ffqs')
    #     # ffqsARI = FFQS.ffqs(data, label, queries=FFQS.queries_cal(
    #     #     datalen), title=title)
    #     # write_ari21(ffqsARI, f'{title}_FFQS')
    #     print('run on ADP')
    #     alpha = 0.22
    #     l = 5
    #     theta = 0.00001
    #     ADPARI = experiment_adp.experiemnt_adp(
    #         data, label, alpha, l, theta, q=datalen)
    #     # os.makedirs('result/ADP', exist_ok=True)
    #     os.makedirs('G:/data/algorithm/mine/p2/result/ADP', exist_ok=True)
    #     pd.DataFrame(ADPARI).to_csv(
    #         f'G:/data/algorithm/mine/p2/result/ADP/{title}.csv', index=False)
    #     ari21ADP = add_ari21(ADPARI, title, ari21ADP)

    #     print('run on ADPE')
    #     xi = [0.6, 0.8]
    #     alpha = [0.15, 0.30]
    #     s = 10
    #     l = 5
    #     theta = 0.00001
    #     ADPEARI = ADPE(data, label, xi, alpha, s, l, theta, q=datalen)
    #     os.makedirs('result/ADPE', exist_ok=True)
    #     os.makedirs('G:/data/algorithm/mine/p2/resultresult/ADPE',
    #                 exist_ok=True)
    #     pd.DataFrame(ADPEARI).to_csv(
    #         f'G:/data/algorithm/mine/p2/resultresult/ADPE/{title}.csv', index=False)
    #     ari21ADPE = add_ari21(ADPEARI, title, ari21ADPE)
    #     # write_ari21(ADPEARI, f'{title}_ADPE')

    #     print('run on cobras')
    #     COBRASARI = COBRAS(data, label, budget=datalen)
    #     # os.makedirs('result/COBRAS', exist_ok=True)
    #     os.makedirs('G:/data/algorithm/mine/p2/result/COBRAS', exist_ok=True)
    #     pd.DataFrame(COBRASARI).to_csv(
    #         f'G:/data/algorithm/mine/p2/result/COBRAS/{title}.csv', index=False)
    #     ari21COBRAS = add_ari21(COBRASARI, title, ari21COBRAS)

    #     pd.DataFrame(ari21ADP).to_csv('ari21ADP.csv', index=False)
    #     pd.DataFrame(ari21ADPE).to_csv('ari21ADPE.csv', index=False)
    #     pd.DataFrame(ari21COBRAS).to_csv('ari21COBRAS.csv', index=False)
    # write_ari21(COBRASARI, f'{title}_COBRAS')

    # print('run on cobra')
    # cobraARI = cobra.experiment(data, label, title)
    # write_ari21(cobraARI, f'{title}_COBRA')

    # pd.DataFrame(ari21ADP).to_csv('ari21ADP.csv', index=False)
    # pd.DataFrame(ari21ADPE).to_csv('ari21ADPE.csv', index=False)
    # pd.DataFrame(ari21COBRAS).to_csv('ari21COBRAS.csv', index=False)
