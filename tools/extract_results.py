import os
import math
import argparse
import numpy as np
from tabulate import tabulate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--res-dir', type=str, default='/home/wxq/od/DeFRCN/checkpoints/cfa/defrcn_gfsod_r101_novel1', help='Path to the results')
    parser.add_argument('--shot-list', type=int, nargs='+', default=[1, 2, 3, 5, 10], help='')
    args = parser.parse_args()

    wf = open(os.path.join(args.res_dir, 'results.txt'), 'w')

    for shot in args.shot_list:

        file_paths = []
        for fid, fname in enumerate(os.listdir(args.res_dir)):
            if fname.split('_')[0] != '{}shot'.format(shot):
                continue
            _dir = os.path.join(args.res_dir, fname)
            if not os.path.isdir(_dir):
                continue
            file_paths.append(os.path.join(_dir, 'log.txt'))
        # 整理好了各个shot的list
        header, results = [], []
        for fid, fpath in enumerate(sorted(file_paths)):
            lineinfos = open(fpath).readlines()
            if fid == 0:
                # -3, -1, -2
                res_info = lineinfos[-2].strip()
                header = res_info.split(':')[-1].split(',')
            res_info = lineinfos[-1].strip()
            try:
                results.append([fid] + [float(x) for x in res_info.split(':')[-1].split(',')])
            except:
                print('[no result]:' + fpath)

        results_np = np.array(results)
        avg = np.mean(results_np, axis=0).tolist()
        cid = [1.96 * s / math.sqrt(results_np.shape[0]) for s in np.std(results_np, axis=0)]
        # 该行代码计算了输入矩阵 results_np 中每个列的 95％ 置信区间的半宽度，用于计算置信区间范围。
        #
        # 具体地，math.sqrt(results_np.shape[0]) 是计算矩阵的行数的平方根，np.std(results_np, axis=0) 则是计算每列的标准差，
        # 而 1.96是标准的 95％ 置信区间下的常数（当样本量足够大时，可使用该常数来计算置信区间）。
        # 最后将计算得到的标准误除以样本量的平方根乘以常数1.96，
        # 即可得到95％ 置信区间的半宽度。所以 cid 是一个包含着每列 95％ 置信区间的半宽度的numpy数组。
        results.append(['μ'] + avg[1:])
        results.append(['c'] + cid[1:])

        table = tabulate(
            results,
            tablefmt="pipe",
            floatfmt=".2f",
            headers=[''] + header,
            numalign="left",
        )

        wf.write('--> {}-shot\n'.format(shot))
        wf.write('{}\n\n'.format(table))
        wf.flush()
    wf.close()

    print('Reformat all results -> {}'.format(os.path.join(args.res_dir, 'results.txt')))


if __name__ == '__main__':
    main()
