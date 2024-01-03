# 用字典存储模型所需参数
# 字典用dict生成，里面key=value
import torch


def get_parameters():
    device_id = 1 if torch.cuda.device_count() > 1 else 0
    cuda = True if torch.cuda.is_available() else False
    device = torch.device(f'cuda:{device_id}' if cuda else "cpu")
    # 训练参数设置
    learning_rate = 1e-3
    early_stop_patients = 200
    batch_size = 16
    epochs = 10000

    heat_width = 48
    heat_height = 48
    sea_surface_width = 24
    sea_surface_height = 24
    cha_width = 144
    cha_height = 144

    input_days = 14
    output_days = 7
    total_feature_count = 7

    # 通过patch_count计算每种输入的patch大小
    patch_count = 36
    patch_heat_width = int(heat_width / (patch_count ** 0.5))
    patch_heat_height = int(heat_height / (patch_count ** 0.5))
    patch_sea_surface_width = int(sea_surface_width / (patch_count ** 0.5))
    patch_sea_surface_height = int(sea_surface_height / (patch_count ** 0.5))
    patch_cha_width = int(cha_width / (patch_count ** 0.5))
    patch_cha_height = int(cha_height / (patch_count ** 0.5))

    d_model = 512
    n_heads = 8
    blocks = 12
    conv_channel = 64

    parameters = dict(cuda=cuda,
                      device=device,
                      learning_rate=learning_rate,
                      early_stop_patients=early_stop_patients,
                      batch_size=batch_size,
                      epochs=epochs,
                      heat_width=heat_width,
                      heat_height=heat_height,
                      sea_surface_width=sea_surface_width,
                      sea_surface_height=sea_surface_height,
                      cha_width=cha_width,
                      cha_height=cha_height,
                      input_days=input_days,
                      output_days=output_days,
                      total_feature_count=total_feature_count,
                      patch_count=patch_count,
                      patch_heat_width=patch_heat_width,
                      patch_heat_height=patch_heat_height,
                      patch_sea_surface_width=patch_sea_surface_width,
                      patch_sea_surface_height=patch_sea_surface_height,
                      patch_cha_width=patch_cha_width,
                      patch_cha_height=patch_cha_height,
                      d_model=d_model,
                      n_heads=n_heads,
                      blocks=blocks,
                      conv_channel=conv_channel
                      )
    return parameters
