import matplotlib.pyplot as plt
import numpy as np
# dice_score = [0.818, 0.821, 0.910, 0.912, 0.898, 0.912, 0.841, 0.916]
infer_time = [4.3, 2.1, 19.79, 12.55, 18.04, 25.86, 21.17, 17.71]
dice_score = [0.398, 0.401, 0.630, 0.654, 0.628, 0.677, 0.516, 0.721]
# infer_time = []

models = ["U-Net", "U-Net++", "Res2Net-50 + PD", "Res2Net-50 + CPD", "PraNet", "HarDMSEG", "EfficientNeteV2-S + PD", "HarDNet-68 + CPD"]
# colors = np.random.rand(8)
# print(colors)
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#d62728']
plt.figure(figsize=(10, 8))
plt.scatter(infer_time, dice_score, s=150, marker="o", c=colors)
# X = np.linspace(0, 30, 30)
# Y = [0.9 for i in range(0, 30)]
# plt.plot(X, Y)
# plt.scatter(17.71, 0.721, s=400, marker='s', c='r')
plt.scatter(17.71, 0.721, s=400, marker='*', c='#d62728')
for i, label in enumerate(models):
    plt.annotate(label, (infer_time[i] - 1, dice_score[i] + 0.008))
    
# plt.title("FPS tương ứng với Meandice trên tập Kvasir-SEG")
plt.xlabel("FPS")
plt.ylabel("Meandice")
plt.xlim([0, 28])
# plt.ylim([0.81, 0.925])
plt.show()  