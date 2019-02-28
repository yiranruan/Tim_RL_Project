import matplotlib.pyplot as plt
import numpy as np

x = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20]  # , 30
x2 = [0, 2, 4, 6, 10, 12, 14, 16, 18]
x3 = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
print(x2)

y = [40.574, 292.696, 385.908, 448.6, 455.708, 459.794,
     468.936, 471.348, 473.736, 479.918, 474.5, 467.872]
success_rate = [12.6, 54.4, 72.1, 83.6, 85.1, 86, 88, 88.9, 89.3, 90.5, 89.9]   # , 89
success_rate2 = [13.1, 60.1, 68.7, 66.1, 70, 72.8, 72.2, 73.4, 72.4]
success_rate3 = [10, 24.8, 42, 50.6, 54.7, 57.2, 63.8, 65.4, 70.2, 73.5, 73.1]


# num = figure number
plt.figure()
plt.plot(x, success_rate, '-o', label='agent 1')
plt.plot(x2, success_rate2, '-o', label='agent 2')
plt.plot(x3, success_rate3, '-o', label='agent 3')

# 取值範圍
plt.xticks(np.arange(0, 21, 2))
plt.ylim((0, 100))

# change label of axis
plt.xlabel('Number of Trainings/100k')
plt.ylabel('Success Rate')

plt.annotate('90.5', xy=(17.5, 92))
plt.annotate('73.4', xy=(15.5, 75))
plt.annotate('73.5', xy=(17.5, 75))

plt.legend()

plt.show()
