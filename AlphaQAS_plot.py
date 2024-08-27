import matplotlib.pyplot as plt
import numpy as np

# 1. 读取TXT文件
filename = './240812_bestParams/Record.txt'  # 假设文件名为 data.txt

with open(filename, 'r') as file:
    lines = file.readlines()

# 初始化数据列表
x_values = []
y1_values = []
y2_values = []
y3_values = []
y4_values = []

y5_values = []
y6_values = []

# 2. 解析数据
for line in lines:
    if line.strip():  # 确保不是空行
        # 按空格分割
        parts = line.split()
        # 提取数据
        x_values.append(float(parts[0]))  # 第0列作为横坐标
        y1_values.append(float(parts[1]))  # 第1列作为第一个y轴数据
        y2_values.append(float(parts[2]))  # 第1列作为第一个y轴数据
        y3_values.append(float(parts[3]))  # 第1列作为第一个y轴数据
        y4_values.append(float(parts[4]))  # 第1列作为第一个y轴数据

        y5_values.append(float(parts[6]))  # 第1列作为第二个y轴数据
        y6_values.append(float(parts[7]))  # 第6列作为第二个y轴数据

# 3. 绘制双y轴图
fig, ax1 = plt.subplots()

# 绘制第一个y轴数据
ax1.plot(x_values, y1_values, 'b--o', label='DNN-mean',linewidth=0.9,markersize = 4)  # 蓝色实线
ax1.plot(x_values, y2_values, 'k--*', label='AlphaSeq-mean',linewidth=0.9,markersize = 4)    # 蓝色实线
ax1.plot(x_values, y3_values, 'b--x', label='DNN-max',linewidth=0.9,markersize = 4)    # 蓝色实线
ax1.plot(x_values, y4_values, 'k--+', label='AlphaSeq-max',linewidth=0.9,markersize = 4)    # 蓝色实线
ax1.set_xlabel('Episode')
ax1.set_ylabel('Metric', color='k')
ax1.tick_params('y', colors='b')

# 添加第二个y轴
ax2 = ax1.twinx()
ax2.plot(x_values, y5_values, 'r--', label='All visited states',linewidth=0.9)  # 红色实线
ax2.plot(x_values, y6_values, 'r--', label='G episode visited states',linewidth=0.9)  # 红色实线
ax2.set_ylabel('No.of visited states', color='r')
ax2.tick_params('y', colors='r')

# 添加图例
ax1.legend(loc='upper left')
# ax2.legend(loc='upper right')

# plt.title('Double Y Axis Plot')
plt.show()



