import numpy as np

patch = np.array([[149, 19, 3],[140, 14, 86],[234, 135, 41]])
template = np.array([[66, 232, 37],[204, 46, 35],[110, 67, 222]])

corr = np.sum(np.multiply(patch, template))

lenght_patch = 0
length_template = 0

for i in range(3):
    for j in range(3):
        lenght_patch += patch[i][j]**2
        length_template += template[i][j]**2

lenght_patch = np.sqrt(lenght_patch)
length_template = np.sqrt(length_template)

result = corr/(lenght_patch*length_template)

print(result)