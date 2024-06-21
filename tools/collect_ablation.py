import csv
import os
from matplotlib import pyplot as plt
import pandas as pd


root = '/home/wxq/od/'
# method_dic = {
#     '0': 'DeFRCN/checkpoints/new_dynamic/2024_05_18_00_00/',
#     '0.1': 'DeFRCN/checkpoints/new_dynamic/2024_06_01_00_00/',
#     '0.3': 'DeFRCN/checkpoints/new_dynamic/2024_06_02_23_00/'
# }
# for method_name, method_path in method_dic.items():
#     for shot in [1, 2, 3, 5, 10]:
#         for split in range(1, 2):
#             txt_file_path = os.path.join(root, method_path, str(split), f'{shot}shot_seed1', 'log.txt')
#             with open(txt_file_path, 'r') as file:
#                 lines = file.readlines()
#
#             # 2. 查找所有含有特定内容的行
#             target_content = "AP,AP50,AP75,bAP,bAP50,bAP75,nAP,nAP50,nAP75"
#             matching_lines = [line for line in lines if target_content in line]
#
#             # 3. 记录这些行下一行中用逗号分开的数字
#             data = []
#             for line in matching_lines:
#                 index = lines.index(line)
#                 next_line = lines[index + 1].strip()
#                 numbers = next_line.split(':')[-1].strip().split(',')
#                 data.append(numbers)
#
#             # 4. 将这些数字以csv形式存储
#             csv_file_path = os.path.join(root, 'ablation_res', f'{method_name}_{split}_{shot}.csv')
#             with open(csv_file_path, 'w', newline='') as csv_file:
#                 csv_writer = csv.writer(csv_file)
#                 csv_writer.writerows(data)


method_dic = {
    '0': 'DeFRCN/checkpoints/new_dynamic/2024_05_18_00_00/',
    '0.1': 'DeFRCN/checkpoints/new_dynamic/2024_06_01_00_00/',
    '0.2': 'DeFRCN/checkpoints/new_dynamic/2024_04_25_23_00/',
    '0.3': 'DeFRCN/checkpoints/new_dynamic/2024_06_02_23_00/'
}

# label = ['0', '0.1', '0.3', '0.2']
# for shot in [1, 2, 3, 5, 10]:
#     print(f"===shot{shot}===")
#     AP = [[], [], [], []]
#     bAP = [[], [], [], []]
#     nAP = [[], [], [], []]
#     max_step = 7
#     for step in range(0, max_step):
#         for i, method_name in enumerate(['0', '0.1', '0.3']):
#             a = pd.read_csv(os.path.join(root, 'ablation_res', f'{method_name}_1_{shot}.csv'), index_col=None, header=None)
#             AP[i].append(a.iloc[step, 1])
#             bAP[i].append(a.iloc[step, 4])
#             nAP[i].append(a.iloc[step, 7])
#
#         a = pd.read_csv(os.path.join(root, 'total_res', f'Ours_1_{shot}.csv'), index_col=None, header=None)
#         AP[3].append(a.iloc[step, 1])
#         bAP[3].append(a.iloc[step, 4])
#         nAP[3].append(a.iloc[step, 7])
#     for i in range(4):
#         print(AP[i])
#         plt.plot([i for i in range(max_step)], AP[i])
#     plt.legend(label)
#     plt.title(f'shot{shot}_AP')
#     plt.show()
#     for i in range(4):
#         plt.plot([i for i in range(max_step)], bAP[i])
#     plt.legend(label)
#     plt.title(f'shot{shot}_bAP')
#     plt.show()
#     for i in range(4):
#         plt.plot([i for i in range(max_step)], nAP[i])
#     plt.legend(label)
#     plt.title(f'shot{shot}_nAP')
#     plt.show()

label = ['0', '0.1', '0.3', '0.2']
for method_name in ['0', '0.1', '0.3']:
    AP = []
    bAP = []
    nAP = []
    for shot in [1, 2, 3, 5, 10]:
        step = 6
        a = pd.read_csv(os.path.join(root, 'ablation_res', f'{method_name}_1_{shot}.csv'), index_col=None, header=None)
        AP.append(a.iloc[step, 1])
        bAP.append(a.iloc[step, 4])
        nAP.append(a.iloc[step, 7])
    print(method_name)
    print(AP)
    print(bAP)
    print(nAP)

AP = []
bAP = []
nAP = []
for shot in [1, 2, 3, 5, 10]:
    step = 6
    a = pd.read_csv(os.path.join(root, 'total_res', f'Ours_1_{shot}.csv'), index_col=None, header=None)
    AP.append(a.iloc[step, 1])
    bAP.append(a.iloc[step, 4])
    nAP.append(a.iloc[step, 7])

print(AP)
print(bAP)
print(nAP)
