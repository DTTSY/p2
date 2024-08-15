from myutil import DataLoader
import os
from sklearn import datasets
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scienceplots
from matplotlib import rcParams

# rcParams['font.family'] = 'serif'
# rcParams['font.serif'] = ['Times New Roman']
rcParams['font.family'] = 'Arial'
# 使用bold字体
# rcParams['font.weight'] = 'bold'


def new_func():
    dataDir = 'G:/data/datasets/UCI/middle/data'
    for f in os.listdir(dataDir):
        print(f'run {f}')
        rdata, real_labels, K, fdata = DataLoader.get_data_from_local(
            f'{dataDir}/{f}', doPerturb=False)
        fdata.to_csv(f'G:/data/datasets/temp/data/{f}', index=False)


def draw():
    # plt.style.use(['science','ieee'])
    x, y = [], []
    z = []
    tt = []
    titles = []
    # d = os.listdir('result\PRDSL-c')
    d = ['tae', 'led', 'divorce', 'wine', 'thyroid', 'ecoli', 'breast', 'musk', 'balance',
         'vehicle', 'banknote', 'segment', 'digits', 'EEG', 'letter', 'avila', 'skin', 'kddcup99']
    # d = ['tae', 'led', 'divorce', 'wine', 'thyroid', 'ecoli', 'breast', 'musk', 'balance',
    #      'vehicle', 'banknote', 'segment']
    # d = ['digits', 'EEG', 'letter', 'avila', 'skin', 'kddcup99']
    d = [f'{f}.csv' for f in d]
    for f in d:
        print(f)
        path = f'result/PRDSL-c2/{f}'
        if not os.path.exists(path):
            continue
        titles.append(f.split('.')[0])
        data = pd.read_csv(path)
        xx = data['human'].values[-1]
        yy = data['c'].values[-1]
        kn = data['expended'].values[-1]
        tt.append(kn)
        # expc = data['expended'].values[-1]
        t = xx + yy
        x.append(xx/kn)
        y.append(yy/kn)
        z.append((kn-t)/kn)
    print(x, y)
    b = np.arange(len(titles))
    bz = np.array(x) + np.array(y)
    # plt.figure(figsize=(12,8))
    plt.bar(b, x, align='center', color='#c95863',
            tick_label=titles, label='human')
    plt.bar(b, y, align='center', bottom=x,
            color='#8da0cb', label='deduction')  # y没写成y1
    plt.bar(b, z, align='center', bottom=bz,
            color='#e6c7b7', label='other')  # y没写成y1
    # 数字标明在柱状图中间

    for xx, yy in zip(b, x):
        plt.text(xx, yy/2, '%.2f' % yy, ha='center', va='bottom', fontsize=10)
    # bz = np.array(z)
    for xx, yy in zip(b, bz):
        plt.text(xx, (yy+1)/2, '%.2f' % float(1-yy),
                 ha='center', va='bottom', fontsize=10)
    for xx, yy in zip(b, tt):
        plt.text(xx, 1, f'{yy}', ha='center', va='bottom', fontsize=10)
    fs = 15
    plt.xlabel('data sets', fontsize=fs)
    plt.xticks(fontsize=fs-2)
    plt.yticks(fontsize=fs-2)
    plt.yscale('log')
    plt.ylabel('query ratio %', fontsize=fs)
    plt.legend(loc='lower right', fontsize=fs)  # 不加这个是没有label显示的！
    # plt.savefig('PRDSL-c.png', dpi=600)
    plt.show()

    # for a, b in zip(b, x):
    #     plt.text(a, b/2, '%.2f' % b, ha='center', va='bottom')
    # fs = 15
    # plt.xlabel('data sets')
    # plt.ylabel('query ratio %')
    # plt.legend(loc='upper right')  # 不加这个是没有label显示的！
    # plt.savefig('PRDSL-c.png', dpi=600)
    # plt.show()


def draw2():
    # plt.style.use(['science', 'ieee'])
    x, y = [], []
    tt = []
    hh = []
    titles = []
    # d = os.listdir('result\PRDSL-c')
    d = ['tae', 'led', 'divorce', 'wine', 'thyroid', 'ecoli', 'breast', 'musk', 'balance',
         'vehicle', 'banknote', 'segment', 'digits', 'EEG', 'letter', 'avila', 'skin', 'kddcup99']
    # d = ['tae', 'led', 'divorce', 'wine', 'thyroid', 'ecoli', 'breast', 'musk', 'balance',
    #      'vehicle', 'banknote', 'segment']
    # d = ['digits', 'EEG', 'letter', 'avila', 'skin', 'kddcup99']
    d = [f'{f}.csv' for f in d]
    for f in d:
        print(f)
        path = f'result/PRDSL-c2/{f}'
        if not os.path.exists(path):
            continue
        titles.append(f.split('.')[0])
        data = pd.read_csv(path)
        xx = data['human'].values[-1]
        yy = data['c'].values[-1]
        t = xx + yy
        hh.append(xx)
        tt.append(t)
        x.append(xx/t)
        y.append(yy/t)
    print(x, y)
    b = np.arange(len(titles))
    plt.figure(figsize=(13, 8))
    plt.bar(b, x, align='center', color='#d14849',
            tick_label=titles, label='Human')
    # plt.bar(b, y, align='center', bottom=x,
    #         color='#5084c3', label='deduction')  # y没写成y1
    plt.bar(b, y, align='center', bottom=x,
            color='#5084c3', label='Deduction')  # y没写成y1
    # 数字标明在柱状图中间
    f1 = 16
    for xx, yy in zip(b, x):
        # 白色字体
        plt.text(xx, .5, '%.3f' % yy, ha='center',
                 va='bottom', fontsize=f1, color='white')
    for i in range(len(b)):
        plt.text(i, .96, '%.3f' % y[i],
                 ha='center', va='bottom', fontsize=f1, color='white')
    # bz = np.array(z)
    for xx, yy in zip(b, tt):
        plt.text(xx, 1, f'{yy}', ha='center', va='bottom', fontsize=13)
    fs = 20
    plt.xlabel('data sets', fontsize=fs)
    # 不显示上边框和右边框
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    # 坐标轴标签旋转45度并且设置x轴最小值与最大值
    plt.xticks(fontsize=fs-2, rotation=30)
    plt.xlim(-.6, 12.5)
    # plt.xticks(fontsize=fs-2, rotation=30)
    plt.yticks(fontsize=fs-2)
    # plt.yscale('log')
    plt.ylabel('pairwise constants ratio', fontsize=fs)
    # 增大legend大小
    plt.legend(loc='upper right', fontsize=20,
               bbox_to_anchor=(1, .2))  # 不加这个是没有label显示的！
    plt.tight_layout()
    # plt.savefig('PRDSL-c1.png', dpi=600)
    plt.show()

    # for a, b in zip(b, x):
    #     plt.text(a, b/2, '%.2f' % b, ha='center', va='bottom')
    # fs = 15
    # plt.xlabel('data sets')
    # plt.ylabel('query ratio %')
    # plt.legend(loc='upper right')  # 不加这个是没有label显示的！
    # plt.savefig('PRDSL-c.png', dpi=600)
    # plt.show()


def get_da():
    dataDir1 = set(os.listdir('result/PRDSL-c'))
    dataDir2 = set(os.listdir('result/PRDSL-c2'))
    return list(dataDir1 - dataDir2)


def draw1():
    # plt.style.use(['science', 'ieee'])
    x, y = [], []
    tt = []
    hh = []
    titles = []
    # d = os.listdir('result\PRDSL-c')
    d = ['tae', 'led', 'divorce', 'wine', 'thyroid', 'ecoli', 'breast', 'musk', 'balance',
         'vehicle', 'banknote', 'segment', 'digits', 'EEG', 'letter', 'avila', 'skin', 'kddcup99']
    # d = ['tae', 'led', 'divorce', 'wine', 'thyroid', 'ecoli', 'breast', 'musk', 'balance',
    #      'vehicle', 'banknote', 'segment']
    # d = ['digits', 'EEG', 'letter', 'avila', 'skin', 'kddcup99']
    d = [f'{f}.csv' for f in d]
    for f in d:
        print(f)
        path = f'result/PRDSL-c2/{f}'
        if not os.path.exists(path):
            continue
        titles.append(f.split('.')[0])
        data = pd.read_csv(path)
        xx = data['human'].values[-1]
        yy = data['expended'].values[-1]
        t = xx + yy
        hh.append(xx)
        tt.append(t)
        x.append(xx/t)
        y.append(yy/t)
    print(x, y)
    b = np.arange(len(titles))
    plt.figure(figsize=(13, 8))
    plt.bar(b, x, align='center', color='#d14849',
            tick_label=titles, label='Minimal Constraint Graph')
    # plt.bar(b, y, align='center', bottom=x,
    #         color='#5084c3', label='deduction')  # y没写成y1
    plt.bar(b, y, align='center', bottom=x,
            color='#5084c3', label='Expanded')  # y没写成y1
    # 数字标明在柱状图中间
    f1 = 16
    for xx, yy in zip(b, x):
        # 白色字体
        plt.text(xx, .02, '%.3f' % yy, ha='center',
                 va='bottom', fontsize=f1, color='white')
    for i in range(len(b)):
        plt.text(i, .5, f'{y[i]:.3}',
                 ha='center', va='bottom', fontsize=f1, color='white')
    # bz = np.array(z)
    for xx, yy in zip(b, tt):
        plt.text(xx, 1, f'{yy}', ha='center', va='bottom', fontsize=13)
    fs = 20
    plt.xlabel('data sets', fontsize=fs)
    # 不显示上边框和右边框
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    # 坐标轴标签旋转45度并且设置x轴最小值与最大值
    plt.xticks(fontsize=fs-2, rotation=30)
    plt.xlim(-.6, 12.5)
    # plt.xticks(fontsize=fs-2, rotation=30)
    plt.yticks(fontsize=fs-2)
    # plt.yscale('log')
    plt.ylabel('pairwise constants ratio', fontsize=fs)
    # 增大legend大小
    plt.legend(loc='upper right', fontsize=20,
               bbox_to_anchor=(1, .9))  # 不加这个是没有label显示的！
    plt.tight_layout()
    # plt.savefig('PRDSL-c1.png', dpi=600)
    plt.show()

    # for a, b in zip(b, x):
    #     plt.text(a, b/2, '%.2f' % b, ha='center', va='bottom')
    # fs = 15
    # plt.xlabel('data sets')
    # plt.ylabel('query ratio %')
    # plt.legend(loc='upper right')  # 不加这个是没有label显示的！
    # plt.savefig('PRDSL-c.png', dpi=600)
    # plt.show()


def sort_data_by_size():
    dataDir = 'G:/data/datasets/UCI/middle - 副本/data'
    s = []
    for f in os.listdir(dataDir):
        title = f.split('.')[0]
        data, label, k, _data = DataLoader.get_data_from_local(
            dataDir + '/' + f)
        s.append((title, len(data)))
    s.sort(key=lambda x: x[1])
    print([i for i, j in s])


def a():
    da = {'tae': 151*3, 'led': 500*10, 'divorce': 170*2, 'wine': 178*3, 'thyroid': 215*3,
          'ecoli': 336*8, 'breast': 699*2, 'musk': 476*2, 'balance': 625*3, 'vehicle': 846*4, }


if __name__ == '__main__':
    # new_func()
    # load kddcup99 dataset
    draw1()
    # sort_data_by_size()
    # draw()
    # print(get_da())
