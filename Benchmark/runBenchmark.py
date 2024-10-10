from myutil import DataLoader

from MinMax import MinMax
from FFQS import FFQS
from ADP.d_experiment import experiment_adp
from COBRAS.experiment.COBRAS import COBRAS
from COBRA import cobra
from ADPE.a import ADPE
from DSL.main import DSL

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
    data, label, k = DataLoader.get_data_from_local(
        dataDir + '/' + file)
    data = data.values

    datalen = len(data)
    datalen = 1_000*len(data)
    # data = (data - data.mean()) / (data.std())

    print(f'{os.getpid()}\t run on {file}')

    print('run on cobra')

    cobraARI = cobra.experiment(data, label, title)
    # cobraPath = 'result/zoutian/COBRA'
    cobraPath = 'result/COBRA'
    os.makedirs(cobraPath, exist_ok=True)
    r = os.listdir(cobraPath)
    # if file not in r:
    cobraPath = f'{cobraPath}/{title}.csv'
    pdf = pd.DataFrame(cobraARI)
    # index 名字改为interaction
    # 给第一列加上名字ari
    pdf.rename(columns={0: 'ari'}, inplace=True)
    pdf.to_csv(cobraPath, index=True, index_label='interaction')
    # cobraARI.to_csv(f'{cobraPath}/{title}.csv',index=False)

    # print(f'run on DSL')
    # DSL_ARI = DSL(data, label, title=title, q=datalen, k=k)
    # DSLPath = 'result/PRDSL-aexpend'
    # os.makedirs(DSLPath, exist_ok=True)
    # pd.DataFrame(DSL_ARI).to_csv(
    #     f'{DSLPath}/{title}.csv', index=False)

    # print('run on MinMax')
    # MinMaxARI = MinMax.minmax(data, label, queries=MinMax.queries_cal(
    #     datalen), title=file.split('.')[0])
    # print('run on ADP')
    # write_ari21(MinMaxARI, f'{title}_MinMax')
    # print('run on ffqs')
    # ffqsARI = FFQS.ffqs(data, label, queries=FFQS.queries_cal(
    #     datalen), title=title)
    # write_ari21(ffqsARI, f'{title}_FFQS')

    # print('run on ADP')
    # alpha = 0.22
    # l = 5
    # theta = 0.00001
    # ADPARI = experiment_adp.experiemnt_adp(
    #     data, label, alpha, l, theta, q=datalen, title=title)
    # # os.makedirs('result/ADP', exist_ok=True)
    # ADPPath = 'result/ADP'
    # os.makedirs(ADPPath, exist_ok=True)
    # pd.DataFrame(ADPARI).to_csv(
    #     f'{ADPPath}/{title}.csv', index=False)
    # ari21ADP = add_ari21(ADPARI, title, ari21ADP)

    # print('run on ADPE')
    # xi = [0.6, 0.8]
    # alpha = [0.15, 0.30]
    # s = 10
    # l = 5
    # theta = 0.00001
    # ADPEARI = ADPE(data, label, xi, alpha, s, l, theta, q=datalen)

    # ADPEPath = 'result/ADPE'
    # os.makedirs(ADPEPath,
    #             exist_ok=True)
    # pd.DataFrame(ADPEARI).to_csv(
    #     f'{ADPEPath}/{title}.csv', index=False)
    # ari21ADPE = add_ari21(ADPEARI, title, ari21ADPE)
    # write_ari21(ADPEARI, f'{title}_ADPE')

    # print('run on cobras')
    # COBRASARI = COBRAS(data, label, budget=datalen)
    # # os.makedirs('result/COBRAS', exist_ok=True)
    # COBRASPath = 'result/COBRAS'
    # os.makedirs(COBRASPath, exist_ok=True)
    # pd.DataFrame(COBRASARI).to_csv(
    #     f'{COBRASPath}/{title}.csv', index=False)
    # ari21COBRAS = add_ari21(COBRASARI, title, ari21COBRAS)


def run_cobra(file: str):
    cobraPath = 'result/COBRA'
    title = file.split('.')[0]
    dataPath = dataDir + '/' + file

    data, label, k = DataLoader.get_data_from_local(dataPath)
    data = data.values
    cobraARI = cobra.experiment(data, label, title)
    # cobraPath = 'result/zoutian/COBRA'
    os.makedirs(cobraPath, exist_ok=True)
    r = os.listdir(cobraPath)
    # if file not in r:
    cobraPath = f'{cobraPath}/{title}.csv'
    pdf = pd.DataFrame(cobraARI)
    # index 名字改为interaction
    # 给第一列加上名字ari
    pdf.rename(columns={0: 'ari'}, inplace=True)
    pdf.to_csv(cobraPath, index=True, index_label='interaction')


def run_adp(file: str):
    title = file.split('.')[0]
    ADPPath = 'result/ADP'
    title = file.split('.')[0]
    dataPath = dataDir + '/' + file

    data, label, k = DataLoader.get_data_from_local(dataPath)
    data = data.values
    datalen = 1_000*len(data)

    alpha = 0.22
    l = 5
    theta = 0.00001
    ADPARI = experiment_adp.experiemnt_adp(
        data, label, alpha, l, theta, q=datalen, title=title)
    # os.makedirs('result/ADP', exist_ok=True)
    os.makedirs(ADPPath, exist_ok=True)
    pd.DataFrame(ADPARI).to_csv(
        f'{ADPPath}/{title}.csv', index=False)


def run_adpe(file: str):
    title = file.split('.')[0]
    ADPEPath = 'result/ADPE'
    title = file.split('.')[0]
    dataPath = dataDir + '/' + file

    data, label, k = DataLoader.get_data_from_local(dataPath)
    data = data.values

    datalen = 1_00*len(data)
    xi = [0.6, 0.8]
    alpha = [0.15, 0.30]
    s = 10
    l = 5
    theta = 0.00001
    ADPEARI = ADPE(data, label, xi, alpha, s, l, theta, q=datalen)
    os.makedirs(ADPEPath,
                exist_ok=True)
    pd.DataFrame(ADPEARI).to_csv(
        f'{ADPEPath}/{title}.csv', index=False)


def run_cobrs(file: str):
    title = file.split('.')[0]
    COBRASPath = 'result/COBRAS'
    title = file.split('.')[0]
    dataPath = dataDir + '/' + file

    data, label, k = DataLoader.get_data_from_local(dataPath)
    datalen = 1_00*len(data)
    data = data.values

    COBRASARI = COBRAS(data, label, budget=datalen)
    # os.makedirs('result/COBRAS', exist_ok=True)
    os.makedirs(COBRASPath, exist_ok=True)
    pd.DataFrame(COBRASARI).to_csv(
        f'{COBRASPath}/{title}.csv', index=False)


def run_alg_i(algName='', args=None) -> None:
    # title = file.split('.')[0]
    # data, label, k = DataLoader.get_data_from_local(
    #     dataDir + '/' + file)
    # data = data.values

    # datalen = len(data)
    # datalen = 1_000*len(data)
    # data = (data - data.mean()) / (data.std())

    print(f'{os.getpid()}\t run on {algName}')

    print('run on cobra')
    if algName == 'cobra':
        files = args['cobra']['data']
        Parallel(n_jobs=len(files))(delayed(run_cobra)(file)
                                    for file in files if file.endswith('.csv'))

    # cobraARI.to_csv(f'{cobraPath}/{title}.csv',index=False)

    # print(f'run on DSL')

    # DSL_ARI = DSL(data, label, title=title, q=datalen, k=k)
    # DSLPath = 'result/PRDSL-aexpend'
    # os.makedirs(DSLPath, exist_ok=True)
    # pd.DataFrame(DSL_ARI).to_csv(
    #     f'{DSLPath}/{title}.csv', index=False)

    # print('run on MinMax')
    # MinMaxARI = MinMax.minmax(data, label, queries=MinMax.queries_cal(
    #     datalen), title=file.split('.')[0])
    # print('run on ADP')
    # write_ari21(MinMaxARI, f'{title}_MinMax')
    # print('run on ffqs')
    # ffqsARI = FFQS.ffqs(data, label, queries=FFQS.queries_cal(
    #     datalen), title=title)
    # write_ari21(ffqsARI, f'{title}_FFQS')

    # print('run on ADP')
    if algName == 'ADP':
        files = args['ADP']['data']
        Parallel(n_jobs=len(files))(delayed(run_adp)(file)
                                    for file in files if file.endswith('.csv'))
        # alpha = 0.22
        # l = 5
        # theta = 0.00001
        # ADPARI = experiment_adp.experiemnt_adp(
        #     data, label, alpha, l, theta, q=datalen, title=title)
        # # os.makedirs('result/ADP', exist_ok=True)
        # ADPPath = 'result/ADP'
        # os.makedirs(ADPPath, exist_ok=True)
        # pd.DataFrame(ADPARI).to_csv(
        #     f'{ADPPath}/{title}.csv', index=False)
    # ari21ADP = add_ari21(ADPARI, title, ari21ADP)

    # print('run on ADPE')
    if algName == 'ADPE':
        files = args['ADPE']['data']
        Parallel(n_jobs=len(files))(delayed(run_adpe)(file)
                                    for file in files if file.endswith('.csv'))
        # xi = [0.6, 0.8]
        # alpha = [0.15, 0.30]
        # s = 10
        # l = 5
        # theta = 0.00001
        # ADPEARI = ADPE(data, label, xi, alpha, s, l, theta, q=datalen)

        # ADPEPath = 'result/ADPE'
        # os.makedirs(ADPEPath,
        #             exist_ok=True)
        # pd.DataFrame(ADPEARI).to_csv(
        #     f'{ADPEPath}/{title}.csv', index=False)
        # ari21ADPE = add_ari21(ADPEARI, title, ari21ADPE)

    # write_ari21(ADPEARI, f'{title}_ADPE')

    # print('run on cobras')
    if algName == 'cobras':
        files = args['cobras']['data']

        Parallel(n_jobs=len(files))(delayed(run_cobrs)(file)
                                    for file in files if file.endswith('.csv'))

        # COBRASARI = COBRAS(data, label, budget=datalen)
        # # os.makedirs('result/COBRAS', exist_ok=True)
        # COBRASPath = 'result/COBRAS'
        # os.makedirs(COBRASPath, exist_ok=True)
        # pd.DataFrame(COBRASARI).to_csv(
        #     f'{COBRASPath}/{title}.csv', index=False)
        # ari21COBRAS = add_ari21(COBRASARI, title, ari21COBRAS)


if __name__ == "__main__":
    dataDir = 'G:/data/algorithm/mine/p2/data'
    dataDir = 'data'
    # data = 'iris.csv'

    # for file in [data]:
    ari21ADP = {'dataset': [], 'ari': [], 'interaction': []}
    ari21ADPE = {'dataset': [], 'ari': [], 'interaction': []}
    ari21COBRAS = {'dataset': [], 'ari': [], 'interaction': []}

    files = os.listdir(dataDir)
    # files = ['Segmentation.csv', 'Waveform-5000-C3.csv', 'OptDigits.csv',
    #          'EEG Eye State.csv', 'Avila.csv', 'Letter Recognition.csv']
    # files = ['Letter Recognition.csv', 'Avila.csv']
    # files = ['avila.csv', 'skin.csv', 'EEG.csv', 'letter.csv', 'balance.csv']
    alg = ['adp', 'adpe', 'cobra', 'cobras']

    # data = {'data': ['Statlog.csv', 'Pen-Based Digits.csv',
    #                  'Online Shoppers.csv', 'codon.csv', 'Dry Bean.csv',
    #                  'HTRU2.csv', 'Letter Recognition.csv', 'Avila.csv', 'adult.csv']}

    data = {'data': ['Statlog.csv', 'Pen-Based Digits.csv']}

    args = {'adp': data, 'adpe': data,
            'cobra': data, 'cobras': data}

    Parallel(n_jobs=len(alg))(delayed(run_alg_i)(algName=algName, args=args)
                              for algName in alg)

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
