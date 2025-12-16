from cmath import sqrt
import numpy as np

# 已知样本及其标签
samples = [[1, 3], [2, 3], [4, 1], [5, 5], [7, 6], [6, 8], [3, 4], [9, 8]]
labels =  [1, 1, 1, 1, 0, 0, 1, 0]

# 待预测样本及其标签
unkonwn = [7, 7]
unkonwn_label = 0

# 计算距离
dists = []
for sample in samples:
    temp_dist = sqrt((sample[0] - unkonwn[0]) ** 2 + (sample[1] - unkonwn[1]) ** 2)
    dists.append(temp_dist)

# 根据距离排序，得到最近邻已知样本
dists_ascending_order = np.argsort(dists)

# 进行 k-nn
count_label_one = 0
count_label_zero = 0
for k in range(1, 9):
    the_k_th_nearest_neighbor_label = labels[dists_ascending_order[k - 1]]
    # 投票
    if the_k_th_nearest_neighbor_label == 1:
        count_label_one += 1
    else:
        count_label_zero += 1
    # 根据投票得到预测结果
    prediction = 1 if count_label_one > count_label_zero else 0
    print(f"When k == {k}, the prediction is: {prediction}")

