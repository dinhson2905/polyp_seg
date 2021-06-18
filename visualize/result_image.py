import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.gridspec as gridspec

# 'cju5hi52odyf90817prvcwg45.png',
# 'cju3ykamdj9u208503pygyuc8.png',
# 'cju5x00l6m5j608503k78ptee.png',
# 'cju5hi52odyf90817prvcwg45.png',
# 'cju8c2rqzs5t80850d0zky5dy.png',   
kvasir_img_path = './data/TestDataset/Kvasir/images/'
kvasir_gt_path = './data/TestDataset/Kvasir/masks/'
kvasir_file = ['cju2hfqnmhisa0993gpleeldd.png',
                'cju3xga12iixg0817dijbvjxw.png',
                'cju5xkwzxmf0z0818gk4xabdm.png',
                'cju6vifjlv55z0987un6y4zdo.png',
                'cju7fbndk2sl608015ravktum.png',
                'cju3uhb79gcgr0871orbrbi3x.png',         
                ]

kvasir_hardpd_path = './results/HarDMSEG/Kvasir/'
kvasir_hardcpd_path = './results/HarDCPD/Kvasir/'
kvasir_prahardnet_path = './results/PraHarDNet/Kvasir/'
kvasir_resnetpd_path = './results/ResNetPD/Kvasir/'
kvasir_resnetcpd_path = './results/ResNetCPD/Kvasir/'
kvasir_effnetpd_path = '/results/EffNetV2SPD/Kvasir'
kvasir_pranet_path = './results/PraNet/Kvasir/'
kvasir_unetxx_path = './results/UNet++/Kvasir/'
kvasir_unet_path = './results/UNet/Kvasir/'
kvasir_effnetv2pd_path = './results/EffNetV2SPD/Kvasir/'

kvasir_img, kvasir_gt = [], []
kvasir_hardpd, kvasir_hardcpd, kvasir_prahardnet = [], [], []
kvasir_resnetpd, kvasir_resnetcpd, kvasir_pranet = [], [], []
kvasir_unet, kvasir_unetxx, kvasir_effnetv2pd  = [], [], []

for x in kvasir_file:
    img_file = kvasir_img_path + x
    gt_file = kvasir_gt_path + x
    result_hardpd_file = kvasir_hardpd_path + x
    result_hardcpd_file = kvasir_hardcpd_path + x
    result_prahardnet_file = kvasir_prahardnet_path + x
    result_resnetpd_file = kvasir_resnetpd_path + x
    result_resnetcpd_file = kvasir_resnetcpd_path + x
    result_pranet_file = kvasir_pranet_path + x
    result_unetxx_file = kvasir_unetxx_path + x
    result_effnetv2pd_file = kvasir_effnetv2pd_path + x
    result_unet_file = kvasir_unet_path + x
    
    kvasir_img.append(img_file)
    kvasir_gt.append(gt_file)
    kvasir_hardpd.append(result_hardpd_file)
    kvasir_hardcpd.append(result_hardcpd_file)
    kvasir_prahardnet.append(result_prahardnet_file)
    kvasir_resnetpd.append(result_resnetpd_file)
    kvasir_resnetcpd.append(result_resnetcpd_file)
    kvasir_pranet.append(result_pranet_file)
    kvasir_unetxx.append(result_unetxx_file)
    kvasir_unet.append(result_unet_file)
    kvasir_effnetv2pd.append(result_effnetv2pd_file)

imgs = [mpimg.imread(x) for x in kvasir_img]
gts = [mpimg.imread(x) for x in kvasir_gt]
hardpd_results = [mpimg.imread(x) for x in kvasir_hardpd]
hardcpd_results = [mpimg.imread(x) for x in kvasir_hardcpd]
prahardnet_results = [mpimg.imread(x) for x in kvasir_prahardnet]
resnetpd_results = [mpimg.imread(x) for x in kvasir_resnetpd]
resnetcpd_results = [mpimg.imread(x) for x in kvasir_resnetcpd]
pranet_results = [mpimg.imread(x) for x in kvasir_pranet]
unet_results = [mpimg.imread(x) for x in kvasir_unet]
unetxx_results = [mpimg.imread(x) for x in kvasir_unetxx]
effnetv2pd_results = [mpimg.imread(x) for x in kvasir_effnetv2pd]

fig = plt.figure()
grid = fig.add_gridspec(nrows=6, ncols=7, wspace=0, hspace=0)

i = 0
for row in range(6):
    for col in range(7):        
        ax = fig.add_subplot(grid[row, col])        

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
            ax.imshow(gts[i])
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
plt.axis('off')
plt.show()
