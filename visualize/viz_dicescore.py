import numpy as np
import matplotlib.pyplot as plt

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

# models = "rescpd, respd, hardcpd"    
models = ['respd', 'rescpd', 'hardpd', 'hardcpd', 'pranet', 'prahardnet', 'effnetv2pd']

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
    elif model == 'prahardnet':
        log_file = './logs/prahardnet_0805.log'
        f = open(log_file, "r")
        raw = f.readlines()
        prahardnet_dice = []
        for x in raw:
            if "meandice" in x:
                mean_dice = x.split(' ')[-1]
                mean_dice = float(mean_dice.replace("\n", ""))
                prahardnet_dice.append(mean_dice)
        x = prahardnet_dice[-1]
        prahardnet_dice.append(x)
        prahardnet_dice.append(x)
        prahardnet_dice.append(x)
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


plt.figure (figsize=(7,5))
x = [i for i in range(99)]

plt.plot(x, respd_dice, label="Res2Net-50 + PD")
plt.plot(x, rescpd_dice, label="Res2Net-50 + CPD")
plt.plot(x, hardpd_dice, label="HarDNet-68 + PD")
plt.plot(x, hardcpd_dice, label="HarDNet-68 + CPD")
plt.plot(x, prahardnet_dice, label="HarDNet-68 + PD + RA")
plt.plot(x, effnetv2pd_dice, label="EfficientNetV2S + PD")
plt.title('Meandice của tập Kvasir-SEG khi xác thực')
plt.xlabel('epoch')
plt.legend()

plt.show()
