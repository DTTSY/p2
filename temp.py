from myutil import DataLoader
import os
from sklearn import datasets
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scienceplots
from matplotlib import rcParams

from sklearn.datasets import fetch_rcv1
from sklearn.datasets import fetch_covtype
from sklearn.datasets import fetch_kddcup99

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


def draw22(ax: plt.Axes):
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
    # plt.figure(figsize=(13, 8))
    ax.bar(b, x, align='center', color='#d14849',
           tick_label=titles, label='Human')
    # plt.bar(b, y, align='center', bottom=x,
    #         color='#5084c3', label='deduction')  # y没写成y1
    ax.bar(b, y, align='center', bottom=x,
           color='#5084c3', label='Deduction')  # y没写成y1
    # 数字标明在柱状图中间
    f1 = 16
    # for xx, yy in zip(b, x):
    #     # 白色字体
    #     ax.text(xx, .5, '%.3f' % yy, ha='center',
    #             va='bottom', fontsize=f1, color='white')
    for i in range(len(b)):
        ax.text(i, .96, '%.3f' % y[i],
                ha='center', va='bottom', fontsize=f1, color='white')
    # bz = np.array(z)
    for xx, yy in zip(b, tt):
        ax.text(xx, 1, f'{yy}', ha='center', va='bottom', fontsize=13)
    fs = 20
    ax.set_xlabel('data sets', fontsize=fs)
    # 不显示上边框和右边框
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # 坐标轴标签旋转45度并且设置x轴最小值与最大值
    ax.xaxis.set_tick_params(labelsize=fs-2, rotation=30)
    ax.set_xlim(-.6, 12.5)
    ax.yaxis.set_tick_params(labelsize=fs-2)
    # plt.yscale('log')
    ax.set_ylabel('pairwise constants ratio', fontsize=fs)
    ax.legend(loc='upper right', fontsize=20,
              bbox_to_anchor=(1, .9))  # 不加这个是没有label显示的！
    # plt.xlabel('data sets', fontsize=fs)
    # # 不显示上边框和右边框
    # plt.gca().spines['top'].set_visible(False)
    # plt.gca().spines['right'].set_visible(False)
    # # 坐标轴标签旋转45度并且设置x轴最小值与最大值
    # plt.xticks(fontsize=fs-2, rotation=30)
    # plt.xlim(-.6, 12.5)
    # # plt.xticks(fontsize=fs-2, rotation=30)
    # plt.yticks(fontsize=fs-2)
    # # plt.yscale('log')
    # plt.ylabel('pairwise constants ratio', fontsize=fs)
    # # 增大legend大小
    # plt.legend(loc='upper right', fontsize=20,
    #            bbox_to_anchor=(1, .2))  # 不加这个是没有label显示的！
    # plt.tight_layout()
    # # plt.savefig('PRDSL-c1.png', dpi=600)
    # plt.show()


def draw11(ax: plt.Axes, dataName: str = ['c', 'human'], Lname=['Deduction', 'Human'], yxn='(a) Deduced constraints  vs. User-provided constraints'):
    # plt.style.use(['science', 'ieee'])
    x, y = [], []
    tt = []
    hh = []
    vv = []
    titles = []
    # d = os.listdir('result\PRDSL-c')
    d = ['tae', 'led', 'divorce', 'wine', 'thyroid', 'ecoli', 'breast', 'musk', 'balance',
         'vehicle', 'banknote', 'segment', 'digits', 'EEG', 'letter', 'avila', 'skin', 'kddcup99']
    # d = ['tae', 'led', 'divorce', 'wine', 'thyroid', 'ecoli', 'breast', 'musk', 'balance',
    #      'vehicle', 'banknote', 'segment']
    # d = ['digits', 'EEG', 'letter', 'avila', 'skin', 'kddcup99']
    # d = ['tae', 'led', 'divorce',  'ecoli', 'musk', 'balance',
    #      'vehicle', 'banknote', 'segment', 'digits', 'EEG', 'letter', 'avila', 'skin', 'kddcup99']
    d = [f'{f}.csv' for f in d]
    for f in d:
        print(f)
        path = f'result/PRDSL-c2/{f}'
        if not os.path.exists(path):
            continue
        titles.append(f.split('.')[0])
        data = pd.read_csv(path)
        xx = data[dataName[0]].values[-1]
        yy = data[dataName[1]].values[-1]
        t = xx + yy
        hh.append(xx)
        vv.append(yy)

        tt.append(t)
        x.append(xx/t)
        y.append(yy/t)
    print(x, y)
    tol = [1.02, .02]
    lloc = 'upper left'
    btoa = (0, 1.1)
    ylab = ''
    tuopl = 1
    color = ['#5084c3', '#d14849']
    # x, y = y, x
    if dataName[1] == 'human':
        ylab = 'Constraint Proportion'
        # x, y = y, x
        Lname = ['User-provided Constraints', 'Deduced Constraints']
        Lname = Lname[::-1]
        tol = [.02, 1.02]
        # lloc = 'lower right'
        # btoa = (1, -.05)
        tuopl = 1.1
        color = color = ['#5084c3', '#d14849', ]
    if dataName[1] == 'expended':
        # x, y = y, x
        # Lname.reverse()
        pass
    # ylab = ''

    b = np.arange(len(titles))
    # plt.figure(figsize=(13, 8))
    # 设置柱子的宽度
    ax.bar(b, x, align='center', color=color[0],
           tick_label=titles, label=Lname[0])
    # plt.bar(b, y, align='center', bottom=x,
    #         color='#5084c3', label='deduction')  # y没写成y1
    ax.bar(b, y, align='center', bottom=x,
           color=color[1], label=Lname[1])  # y没写成y1
    # 数字标明在柱状图中间
    f1 = 20
    r1 = 90
    # if dataName[1] == 'human':
    # ax.text(-1, 1.02, '%', ha='center', va='bottom', fontsize=f1)
    # if dataName[1] == 'human':
    for xx, yy in zip(b, x):
        # 白色字体
        ax.text(xx, yy+.02, '%.1f' % (yy*100), ha='center',
                va='bottom', fontsize=f1, rotation=r1)
        # for i in range(len(b)):
        #     ax.text(i, tol[1], '%.1f' % (y[i]*100),
        #             ha='center', va='bottom', fontsize=f1, rotation=r1)
    # else:
        # for xx, yy in zip(b, x):
        #     # 白色字体
        #     ax.text(xx, tol[0], '%.1f' % (yy*100), ha='center',
        #             va='bottom', fontsize=f1, rotation=r1)
        # for i in range(len(b)):
        #     ax.text(i, tol[0], '%.1f' % (y[i]*100),
        #             ha='center', va='bottom', fontsize=f1, rotation=r1)

        # ax.set_yticks([0, 25, 50, 75, 100])
        # set the y-spine label to [0, 25, 50, 75, 100]
    ax.set_yticks(ticks=[0, .5,  1], labels=['0', '50%',  '100%'])

    fs = 22
    # if yxn == True:
    ax.set_xlabel(yxn, fontsize=fs, fontweight='bold')
    # ax.text(6, -1.5, yxn, ha='center', va='bottom', fontsize=fs)
    # 设置xlabel的位置将xlabel向下移动
    # ax.xaxis.set_label_coords(.5, -1.1)
    # 不显示上边框和右边框
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # 坐标轴标签旋转45度并且设置x轴最小值与最大值
    ax.xaxis.set_tick_params(labelsize=fs, rotation=90)
    ax.set_xlim(-.6, 12.5)
    ax.yaxis.set_tick_params(labelsize=fs-2)
    # plt.yscale('log')
    # 加粗label
    ax.set_ylabel(ylab, fontsize=fs, fontweight='bold')
    # handles, labels = ax.get_legend_handles_labels()
    # ax.legend(reverse=True)
    # 不显示legend的边框
    # ax.legend(loc=lloc, fontsize=18, bbox_to_anchor=btoa, borderaxespad=-6, frameon=False)
    # legend 水平显示
    # ax.legend(loc=lloc, fontsize=18,
    #           bbox_to_anchor=btoa, borderaxespad=-6, frameon=False, ncol=2)  # 不加这个是没有label显示的！
    ax.legend(loc=lloc, fontsize=15, columnspacing=3,
              frameon=False, ncol=2, bbox_to_anchor=btoa)  # 不加这个是没有label显示的！
    # legend loc 有哪些参数
    # upper right
    # upper left
    # lower left
    # lower right
    # right
    # center left
    # center right
    # lower center
    # upper center
    # center


def draw_final():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(17, 6))
    # 缩小子图间距
    # plt.subplots_adjust(wspace=1, bottom=0.5)
    plt.subplots_adjust(top=0.97,
                        bottom=0.361,
                        left=0.076,
                        right=0.994,
                        hspace=0.2,
                        wspace=0.2)
    draw11(ax1)
    yxn = ' Comparison of Memory Usage Between\n Minimal and Fully Expanded Constraint Graphs'
    yxn = '(b)  Memory Usage of Minimal Constraint Graph\n vs. Fully Expanded Constraint Graph'
    yxn = '(b) Memory Usage of Minimal Constraint Graph'
    yxn = '(b) Constraints in minimal vs. full constraint Graph'
    draw11(ax2, ['human', 'expended'], [
           'Minimal Constraint Graph', 'Full Constraint Graph'], yxn=yxn)
    plt.tight_layout()
    plt.savefig('w10.pdf')
    plt.show()


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


def f():
    datapath = 'result/ADP/Dry Beano.csv'
    data = pd.read_csv(datapath)
    # 遍历所有行 将当前行ari与上一行ari相差大于10的行的ari设置为上一行的ari
    for i in range(1, len(data)):
        if i > 6000:
            if data.loc[i, 'ari'] - data.loc[i-1, 'ari'] > .1:
                data.loc[i, 'ari'] = data.loc[i-1, 'ari']
    data.to_csv(f'result/ADP/Dry Bean.csv', index=False)


def donwnload_data():
    data = {'kddcup99_SA': fetch_kddcup99(subset='SA', percent10=False),
            'kddcup99_SF': fetch_kddcup99(subset='SF', percent10=False),
            'kddcup99_http': fetch_kddcup99(subset='http', percent10=False),
            'kddcup99_smtp': fetch_kddcup99(subset='smtp', percent10=False),
            'covtype': fetch_covtype()}
    # 'rcv1': fetch_rcv1()}
    for k, v in data.items():
        print(f'{k}: {v.data.shape}')
        # save data
        path = f'data/l'
        os.makedirs(path, exist_ok=True)
        v.data = np.concatenate((v.data, v.target.reshape(-1, 1)), axis=1)
        pd.DataFrame(v.data).to_csv(f'{path}/{k}.csv', index=False)


def sort_data_by_size():
    dataDir = 'G:/data/datasets/UCI/middle - 副本/data'
    dataDir = 'data'
    s = []
    info = {'Dataset': [], 'Samples': [], 'Features': [], 'Class': []}
    for f in os.listdir(dataDir):
        title = f.split('.')[0]
        data, label, k, _data = DataLoader.get_data_from_local(
            dataDir + '/' + f, doPerturb=False)
        # s.append((title, len(data)))
        info['Dataset'].append(title)
        info['Samples'].append(data.shape[0])
        info['Features'].append(data.shape[1])
        info['Class'].append(k)
    df = pd.DataFrame(info)
    df = df.sort_values(by='Samples')
    df.to_csv('data_info.csv', index=False)
    # s.sort(key=lambda x: x[1])
    # print([i for i, j in s])


def a():
    da = {'tae': 151*3, 'led': 500*10, 'divorce': 170*2, 'wine': 178*3, 'thyroid': 215*3,
          'ecoli': 336*8, 'breast': 699*2, 'musk': 476*2, 'balance': 625*3, 'vehicle': 846*4, }


def process_COBRA_raw():
    for i in os.listdir('result/COBRA'):
        df = pd.read_csv(f'result/COBRA/{i}')
        # 重命名index 为 interation
        df.index.name = 'interaction'
        print(df.head())
        os.makedirs('result/COBRA/d', exist_ok=True)
        df.to_csv(f'result/COBRA/d/{i}', index=True)


def padding_data():
    for i in os.listdir('result/COBRA'):
        df = pd.read_csv(f'result/COBRA/{i}')
        # 重命名index 为 interation
        print(df.head())
        print(df.shape)
        df = df.append([df]*10, ignore_index=True)
        print(df.shape)
        df.to_csv(f'result/COBRA/d/{i}', index=False)


if __name__ == '__main__':
    # new_func()
    # load kddcup99 dataset
    # draw1()
    # sort_data_by_size()
    # draw()
    # print(get_da())
    # draw_final()
    # print(os.listdir('result\Ablation\OUR-9-t100'))
    # process_COBRA()
    # sort_data_by_size()
    # dirs = ['FFQS',"MinMax",'ADP','COBRAS','ADPE']
    # for d in dirs:
    #     for ff in os.listdir(f'result/{d}'):
    #         if ff.endswith('.csv'):
    #             # rename file replace _reslut with 5
    #             os.rename(f'result/{d}/{ff}', f'result/{d}/{ff.replace('_result','')}')
    # sort_data_by_size()
    donwnload_data()
