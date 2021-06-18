import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.gridspec as gridspec
# 103 1 11 110 12 123 71 13 139 14 148 151 96 89
etis_img_path = './data/TestDataset/ETIS-LaribPolypDB/images/'
etis_gt_path = './data/TestDataset/ETIS-LaribPolypDB/masks/'
etis_file = ['103.png', '1.png', '110.png', '12.png', '14.png', '151.png']

etis_hardpd_path = './results/HarDMSEG/ETIS-LaribPolypDB/'
etis_hardcpd_path = './results/HarDCPD/ETIS-LaribPolypDB/'
etis_prahardnet_path = './results/PraHarDNet/ETIS-LaribPolypDB/'
etis_resnetpd_path = './results/ResNetPD/ETIS-LaribPolypDB/'
etis_resnetcpd_path = './results/ResNetCPD/ETIS-LaribPolypDB/'
etis_effnetpd_path = '/results/EffNetV2SPD/ETIS-LaribPolypDB'
etis_pranet_path = './results/PraNet/ETIS-LaribPolypDB/'
etis_unetxx_path = './results/UNet++/ETIS-LaribPolypDB/'
etis_unet_path = './results/UNet/ETIS-LaribPolypDB/'
etis_effnetv2pd_path = './results/EffNetV2SPD/ETIS-LaribPolypDB/'

etis_img, etis_gt = [], []
etis_hardpd, etis_hardcpd, etis_prahardnet = [], [], []
etis_resnetpd, etis_resnetcpd, etis_pranet = [], [], []
etis_unet, etis_unetxx, etis_effnetv2pd  = [], [], []

for x in etis_file:
    img_file = etis_img_path + x
    gt_file = etis_gt_path + x
    result_hardpd_file = etis_hardpd_path + x
    result_hardcpd_file = etis_hardcpd_path + x
    result_prahardnet_file = etis_prahardnet_path + x
    result_resnetpd_file = etis_resnetpd_path + x
    result_resnetcpd_file = etis_resnetcpd_path + x
    result_pranet_file = etis_pranet_path + x
    result_unetxx_file = etis_unetxx_path + x
    result_effnetv2pd_file = etis_effnetv2pd_path + x
    result_unet_file = etis_unet_path + x
    
    etis_img.append(img_file)
    etis_gt.append(gt_file)
    etis_hardpd.append(result_hardpd_file)
    etis_hardcpd.append(result_hardcpd_file)
    etis_prahardnet.append(result_prahardnet_file)
    etis_resnetpd.append(result_resnetpd_file)
    etis_resnetcpd.append(result_resnetcpd_file)
    etis_pranet.append(result_pranet_file)
    etis_unetxx.append(result_unetxx_file)
    etis_unet.append(result_unet_file)
    etis_effnetv2pd.append(result_effnetv2pd_file)

imgs = [mpimg.imread(x) for x in etis_img]
gts = [mpimg.imread(x) for x in etis_gt]
hardpd_results = [mpimg.imread(x) for x in etis_hardpd]
hardcpd_results = [mpimg.imread(x) for x in etis_hardcpd]
prahardnet_results = [mpimg.imread(x) for x in etis_prahardnet]
resnetpd_results = [mpimg.imread(x) for x in etis_resnetpd]
resnetcpd_results = [mpimg.imread(x) for x in etis_resnetcpd]
pranet_results = [mpimg.imread(x) for x in etis_pranet]
unet_results = [mpimg.imread(x) for x in etis_unet]
unetxx_results = [mpimg.imread(x) for x in etis_unetxx]
effnetv2pd_results = [mpimg.imread(x) for x in etis_effnetv2pd]

fig = plt.figure(figsize=(11, 10))
grid = fig.add_gridspec(nrows=6, ncols=7, hspace=0.2)

i = 0
for row in range(6):
    for col in range(7):        
        ax = fig.add_subplot(grid[row, col])
        ax.set(xticks=[], yticks=[])
        
        if row == 0:
            if col == 0:
                ax.set_title("Images")
            elif col == 1:
                ax.set_title("Groundtruth")
            elif col == 2:
                ax.set_title("HarDNet + CPD")
            elif col == 3:
                ax.set_title("HarDMSEG")
            elif col == 4:
                ax.set_title("PraNet")
            elif col == 5:
                ax.set_title("UNet++")
            elif col == 6:
                ax.set_title("UNet")
        if col == 0:
            ax.imshow(imgs[i])
        elif col == 1:
            ax.imshow(gts[i], cmap='gray')
        elif col == 2:
            ax.imshow(hardcpd_results[i], cmap='gray')
        elif col == 3:
            ax.imshow(hardpd_results[i], cmap='gray') 
        elif col == 4:
            ax.imshow(pranet_results[i], cmap='gray')
        elif col == 5:
            ax.imshow(unetxx_results[i], cmap='gray')
        elif col == 6:
            ax.imshow(unet_results[i], cmap='gray')
    i += 1
# plt.colorbar()
plt.show()
