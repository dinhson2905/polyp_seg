from logs import setup_logger
log_path = 'logs/'
file_hardpd = log_path + 'eval_HarDPD.log'
file_hardcpd = log_path + 'eval_HarDCPD.log'
file_pranet = log_path + 'eval_PraNet.log'
file_unet = log_path + 'eval_UNet.log'
# file_hardpd = log_path + 'eval_HarDPD.log'
# file_hardpd = log_path + 'eval_HarDPD.log'
# file_hardpd = log_path + 'eval_HarDPD.log'
# file_hardpd = log_path + 'eval_HarDPD.log'
with open(file_hardpd, "r") as f:
    hardpd = f.readlines()
    f.close()

with open(file_hardcpd, "r") as f:
    hardcpd = f.readlines()
    f.close()

with open(file_pranet, "r") as f:
    pranet = f.readlines()
    f.close()

with open(file_unet, "r") as f:
    unet = f.readlines()
    f.close()
for i in range(20):
    unet.append("\n")

log_file = 'logs/atsume.log'
logger = setup_logger('eval_logger', log_file)

for a1, a2, a3, a4 in zip(hardpd, hardcpd, pranet, unet):
    # if a4 == None:
    #     a4 = "\n"
    a2 = a2.split(" ")[-1]
    a3 = a3.split(" ")[-1]
    a4 = a4.split(" ")[-1]
    x = a1.replace("\n", " ") + a2.replace("\n", " ") + a3.replace("\n", " ") + a4.replace("\n", " ")
    
    logger.info(x)

    