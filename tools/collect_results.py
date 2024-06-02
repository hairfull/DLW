import csv

for shot in [1, 2, 3, 5, 10]:
    for split in range(1, 4):
        txt_file_path = f'/home/wxq/od/DeFRCN/checkpoints/rdd1/defrcn_gfsod_r101_novel{split}/spcb/{shot}shot_seed1/'
        with open(txt_file_path + 'log.txt', 'r') as file:
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
        csv_file_path = txt_file_path + 'output.csv'
        with open(csv_file_path, 'w', newline='') as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerows(data)

