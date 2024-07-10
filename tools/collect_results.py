import csv
import os
import pandas as pd

root = '/home/wxq/od/'
method_dic = {
    'Ours' : 'DeFRCN/checkpoints/new_dynamic/2024_04_25_23_00/',
    'Upsample' : 'DeFRCN/checkpoints/new_upsample/2024_05_07_13_00/',
    # 'DeFRCN' : 'DeFRCN/checkpoints/defrcn_ori/',
    # 'FSRDD' : 'DeFRCN/checkpoints/spcb/',
    # 'CFA' : 'DeFRCN/checkpoints/cfa/',
    # 'TFA-cos' : 'few-shot-object-detection/checkpoints/rdd/',
    # 'TFA-fc' : 'few-shot-object-detection/checkpoints/rdd-fc/'
}
for method_name, method_path in method_dic.items():
    for shot in [1, 2, 3, 5, 10]:
        for split in range(1, 4):
            txt_file_path = os.path.join(root, method_path, str(split), f'{shot}shot_seed1', 'log.txt')
            with open(txt_file_path, 'r') as file:
                lines = file.readlines()

            # 2. 查找所有含有特定内容的行
            target_content = "AP,AP50,AP75,bAP,bAP50,bAP75,nAP,nAP50,nAP75"
            matching_lines = [line for line in lines if target_content in line]

            # 3. 记录这些行下一行中用逗号分开的数字
            data = []
            for line in matching_lines:
                index = lines.index(line)
                next_line = lines[index + 1].strip()
                numbers = next_line.split(':')[-1].strip().split(',')
                data.append(numbers)

            # 4. 将这些数字以csv形式存储
            csv_file_path = os.path.join(root, 'total_res', f'{method_name}_{split}_{shot}.csv')
            with open(csv_file_path, 'w', newline='') as csv_file:
                csv_writer = csv.writer(csv_file)
                csv_writer.writerows(data)

for method_name, method_path in method_dic.items():
    for shot in [5]:
        res = [0, 0, 0, 0, 0, 0, 0, 0, 0]
        for split in range(1, 4):
            a = pd.read_csv(os.path.join(root, 'total_res', f'{method_name}_{split}_{shot}.csv'), index_col=None, header=None)
            # print(res)
            print(f'{method_name} {a.iloc[6, 2]} {a.iloc[6, 5]} {a.iloc[6, 8]}')
        # print(f'{method_name} {[r / 3 for r in res]}')
