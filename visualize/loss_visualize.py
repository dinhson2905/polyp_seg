import numpy as np
import matplotlib.pyplot as plt

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

models = ["rescpd", "respd", "hardpd", "hardcpd", "effnetv2pd"]    

for model in models:
    if model == 'respd':
        log_file = './logs/resnetpd_2021052510.log'
        f = open(log_file, "r")
        raw = f.readlines()
        loss_respd = []
        for x in raw:
            if "Loss" in x:
                x = x.split(' ')[-1]
                x = float(x.split('\n')[0])
                loss_respd.append(x)
        loss_respd = list(chunks(loss_respd, 19))
        for i, l in enumerate(loss_respd):
            m = np.mean(l)
            loss_respd[i] = m

    elif model == 'hardpd':
        log_file = './logs/hardpd_0611.log'
        f = open(log_file, "r")
        raw = f.readlines()
        loss_hardpd = []
        for x in raw:
            if "Loss" in x:
                x = x.split(' ')[-1]
                x = float(x.split('\n')[0])
                loss_hardpd.append(x)
        loss_hardpd = list(chunks(loss_hardpd, 5))
        for i, l in enumerate(loss_hardpd):
            m = np.mean(l)
            loss_hardpd[i] = m
    
    elif model == 'effnetv2pd':
        log_file = './logs/effnetv2pd.log'
        f = open(log_file, "r")
        raw = f.readlines()
        loss_effnetv2pd = []
        for x in raw:
            if "Loss" in x:
                x = x.split(' ')[-1]
                x = float(x.split('\n')[0])
                loss_effnetv2pd.append(x)
        loss_effnetv2pd = list(chunks(loss_effnetv2pd, 5))
        for i, l in enumerate(loss_effnetv2pd):
            m = np.mean(l)
            loss_effnetv2pd[i] = m

    elif model == 'rescpd':
        log_file = './logs/resnetcpd_2021052802.log'
        f = open(log_file, "r")
        raw = f.readlines()
        loss_attention = []
        loss_detection = []
        loss_total_rescpd = []
        for x in raw:
            if "Loss" in x:        
                x1 = x.split(' ')[-2]
                x1 = float(x1.replace(",", "").replace("[", ""))
                loss_attention.append(x1)
                x2 = x.split(' ')[-1]
                x2 = float(x2.replace("]", "").replace("\n", ""))
                loss_detection.append(x2)
        for loss1, loss2 in zip(loss_attention, loss_detection):
            loss_total = loss1 + loss2
            loss_total_rescpd.append(loss_total)
        
        loss_total_rescpd = list(chunks(loss_total_rescpd, 5))
        for i, l in enumerate(loss_total_rescpd):
            m = np.mean(l)
            loss_total_rescpd[i] = m

    elif model == 'hardcpd':
        log_file = './logs/hardcpd1104.log'
        f = open(log_file, "r")
        raw = f.readlines()
        loss_attention = []
        loss_detection = []
        loss_total_hardcpd = []
        for x in raw:
            if "Loss" in x:
                x1 = x.split(' ')[-3]
                x1 = float(x1.replace(",", ""))
                loss_attention.append(x1)
                x2 = x.split(' ')[-1]
                x2 = float(x2.replace("]", "").replace("\n", ""))
                loss_detection.append(x2)
        for loss1, loss2 in zip(loss_attention, loss_detection):
            loss_total = loss1 + loss2
            loss_total_hardcpd.append(loss_total)
        loss_total_hardcpd = list(chunks(loss_total_hardcpd, 5))
        for i, l in enumerate(loss_total_hardcpd):
            m = np.mean(l)
            loss_total_hardcpd[i] = m

for model in models:
    if model == 'respd':
        log_file = './logs/resnetpd_2021052510.log'
        f = open(log_file, "r")
        raw = f.readlines()
        respd_dice = []
        for x in raw:
            if "meandice" in x:
                mean_dice = x.split(' ')[-1]
                mean_dice = float(mean_dice.replace("\n", ""))
                respd_dice.append(mean_dice)
    elif model == 'hardpd':
        log_file = './logs/hardpd_0611.log'
        f = open(log_file, "r")
        raw = f.readlines()
        hardpd_dice = []
        for x in raw:
            if "meandice" in x:
                mean_dice = x.split(' ')[-1]
                mean_dice = float(mean_dice.replace("\n", ""))
                hardpd_dice.append(mean_dice)

    elif model == 'effnetv2pd':
        log_file = './logs/effnetv2pd.log'
        f = open(log_file, "r")
        raw = f.readlines()
        effnetv2pd_dice = []
        for x in raw:
            if "meandice" in x:
                mean_dice = x.split(' ')[-1]
                mean_dice = float(mean_dice.replace("\n", ""))
                effnetv2pd_dice.append(mean_dice)

    elif model == 'rescpd':
        log_file = './logs/resnetcpd_2021052802.log'
        f = open(log_file, "r")
        raw = f.readlines()
        rescpd_dice = []
        for x in raw:
            if "meandice" in x:
                mean_dice = x.split(' ')[-1]
                mean_dice = float(mean_dice.replace("\n", ""))
                rescpd_dice.append(mean_dice)
    elif model == 'hardcpd':
        log_file = './logs/hardcpd_1204.log'
        f = open(log_file, "r")
        raw = f.readlines()
        hardcpd_dice = []
        for x in raw:
            if "meandice" in x:
                mean_dice = x.split(' ')[-1]
                mean_dice = float(mean_dice.replace("\n", ""))
                hardcpd_dice.append(mean_dice)

plt.figure(figsize=(8, 5))
# plt.subplot(121)
# x = [i for i in range(99)]
# plt.plot(x, loss_respd, label="ResNet-50 + PD", color="b")
# plt.plot(x, loss_total_rescpd, label="ResNet-50 + CPD", color="g")
# plt.plot(x, loss_hardpd, label="HarDNet-68 + PD", color="r")
# plt.plot(x, loss_total_hardcpd, label="HarDNet-68 + CPD", color="c")
# plt.plot(x, loss_effnetv2pd, label="EfficientNetV2S + PD", color="m")
# plt.title("Đồ thị hàm mất mát")
# plt.xlabel("epoch")
# plt.legend()

# plt.subplot(122)
x = [i for i in range(99)]

plt.plot(x, respd_dice, label="Res2Net-50 + PD", color="b")
plt.plot(x, rescpd_dice, label="Res2Net-50 + CPD", color="g")
plt.plot(x, hardpd_dice, label="HarDNet-68 + PD", color="r")
plt.plot(x, hardcpd_dice, label="HarDNet-68 + CPD", color="c")
plt.plot(x, effnetv2pd_dice, label="EfficientNetV2S + PD", color="m")
plt.title('Đồ thị Meandice của tập xác thực Kvasir-SEG')
plt.xlabel('epoch')
plt.legend()
plt.axis('off')
plt.show()
