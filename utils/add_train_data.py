import random

# 读取train_data.txt文件并分割样本
one_nine_samples = []
two_eight_samples = []
three_seven_samples = []
undeal_samples = []

with open('../data/train_data.txt', 'r') as file:
    for line in file:
        outfit_id, skc1, skc2, score = line.strip().split(' ')
        score = int(score)
        if score in [9, 10]:
            one_nine_samples.append(line)
            one_nine_samples.append(outfit_id + ' ' + str(skc2) + ' ' + str(skc1) + ' ' + str(score))
        elif score in [1, 2, 3, 8]:
            two_eight_samples.append(line)
            two_eight_samples.append(outfit_id + ' ' + str(skc2) + ' ' + str(skc1) + ' ' + str(score))
        elif score in [4, 7]:
            three_seven_samples.append(line)
            three_seven_samples.append(outfit_id + ' ' + str(skc2) + ' ' + str(skc1) + ' ' + str(score))
        else:
            undeal_samples.append(line)
            undeal_samples.append(outfit_id + ' ' + str(skc2) + ' ' + str(skc1) + ' ' + str(score))

# 复制样本
one_nine_samples = one_nine_samples * 6
two_eight_samples = two_eight_samples * 4
three_seven_samples = three_seven_samples * 2

# 合并样本列表
new_samples = one_nine_samples + two_eight_samples + three_seven_samples + undeal_samples

# 打乱样本顺序
random.shuffle(new_samples)

# 写入更新后的train_data.txt文件
with open('../data/train_data_add.txt', 'w') as file:
    for sample in new_samples:
        file.write(sample + '\n')

# 读取train_data.txt文件，并删除空行
with open('../data/train_data_add.txt', 'r') as file:
    lines = file.readlines()

# 去除每行末尾的换行符，并过滤掉空行
lines = [line.strip() for line in lines if line.strip()]

# 将处理后的内容写入回train_data.txt文件
with open('../data/train_data_add.txt', 'w') as file:
    file.write('\n'.join(lines))
