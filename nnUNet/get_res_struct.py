import torch
from torchsummary import summary

# 直接加载整个模型
checkpoint = torch.load('/home/liushuo/isensee/temp/nnInteractive_v1.0/fold_0/checkpoint_final.pth', map_location='cpu')
# model.eval()

# 打印模型结构
# print(model)

# 1. 加载整个 checkpoint


# 2. 从 checkpoint 里取出真正的 state_dict（这里假设 key 名叫 'network_weights'）
#    如果你的 checkpoint 结构不一样，请改成 checkpoint['xxx'] 对应正确的字段名
full_state_dict = checkpoint['network_weights']  

# 3. 筛选出 encoder 部分
encoder_state_dict = {
    k[len('encoder.'):]: v
    for k, v in full_state_dict.items()
    if k.startswith('encoder.')
}

# 4. 筛选出 decoder 部分
decoder_state_dict = {
    k[len('decoder.'):]: v
    for k, v in full_state_dict.items()
    if k.startswith('decoder.')
}
# 假设输入是 3 通道、224×224 的图像
# summary(model, input_size=(3, 224, 224), device='cpu')
# 2) 分别保存 encoder 和 decoder 的参数
# torch.save(encoder_state_dict, '/home/liushuo/isensee/temp/nnInteractive_v1.0/fold_0/encoder_only.pth')
# torch.save(decoder_state_dict, '/home/liushuo/isensee/temp/nnInteractive_v1.0/fold_0/decoder_only.pth')