from .swin import SwinTransformer
import configparser

def build_model(config):
    img_size = int(config['DATA']['IMG_SIZE'])
    patch_size = int(config['SWIN']['PATCH_SIZE'])
    in_chans = int(config['SWIN']['IN_CHANS'])
    num_classes = int(config['SWIN']['NUM_CLASSES'])
    embed_dim = int(config['SWIN']['EMBED_DIM'])
    depths = config['SWIN']['DEPTHS']
    depths = depths.replace("[", "")
    depths = depths.replace("]", "")
    depths = list(depths.split(","))
    depths = [int(x) for x in depths]
    num_heads = config['SWIN']['NUM_HEADS']
    num_heads = num_heads.replace("[", "")
    num_heads = num_heads.replace("]", "")
    num_heads = list(num_heads.split(","))
    num_heads = [int(x) for x in num_heads]
    window_size = int(config['SWIN']['WINDOW_SIZE'])
    mlp_ratio = float(config['SWIN']['MLP_RATIO'])
    qkv_bias = config['SWIN'].getboolean('QKV_BIAS')
    qk_scale = config['SWIN']['QK_SCALE']
    if qk_scale == "None":
        qk_scale = None
    else:
        qk_scale = float(qk_scale)
    drop_rate = float(config['MODEL']['DROP_RATE'])
    drop_path_rate = float(config['MODEL']['DROP_PATH_RATE'])
    ape = config['SWIN'].getboolean('APE')
    patch_norm = config['SWIN'].getboolean('PATCH_NORM')
    use_checkpoint = config['TRAIN'].getboolean('USE_CHECKPOINT')

    # print(f'{img_size} {patch_size} {in_chans} {num_classes} {embed_dim} {depths} {num_heads} {window_size} {mlp_ratio} {qk_scale} {qkv_bias} {drop_rate} {drop_path_rate} {ape} {use_checkpoint} {patch_norm}')

    
    model = SwinTransformer(img_size=img_size,
                                patch_size=patch_size,
                                in_chans=in_chans,
                                num_classes=num_classes,
                                embed_dim=embed_dim,
                                depths=depths,
                                num_heads=num_heads,
                                window_size=window_size,
                                mlp_ratio=mlp_ratio,
                                qkv_bias=qkv_bias,
                                qk_scale=qk_scale,
                                drop_rate=drop_rate,
                                drop_path_rate=drop_path_rate,
                                ape=ape,
                                patch_norm=patch_norm,
                                use_checkpoint=use_checkpoint)
    return model
    
