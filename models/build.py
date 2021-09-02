from .swin_transformer import SwinTransformer

def build_model(config):
    model_type = config.MODEL.TYPE
    if model_type == 'swin':
        model = SwinTransformer(
            img_size=config.DATA.IMG_SIZE,
            patch_size=config.MODEL.SWIN.PATCH_SIZE,
            in_channels=config.MODEL.SWIN.IN_CHANNELS,
            num_classes=config.MODEL.NUM_CLASSES,
            embed_dim=config.MODEL.SWIN.EMBED_DIM,
            depths=config.MODEL.SWIN.DEPTHS,
            num_heads=config.MODEL.SWIN.NUM_HEADS,
            window_size=config.MODEL.SWIN.WINDOW_SIZE,
            mlp_ratio=config.MODEL.SWIN.MLP_RATIO,
            qkv_bias=config.MODEL.SWIN.QKV_BIAS,
            qk_scale=config.MODEL.SWIN.QK_SCALE,
            drop_rate=config.MODEL.DROP_RATE,
            drop_path_rate=config.MODEL.DROP_PATH_RATE,
            patch_norm=config.MODEL.SWIN.PATCH_NORM
        )
    else:
        raise NotImplementedError("Unknown model: '{}'".format(model_type))

    return model