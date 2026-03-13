import torch

enc_sd = torch.load("/home/liushuo/isensee/temp/nnInteractive_v1.0/fold_0/encoder_only.pth", map_location="cpu")
dec_sd = torch.load("/home/liushuo/isensee/temp/nnInteractive_v1.0/fold_0/decoder_only.pth", map_location="cpu")
print("ENC keys sample:", list(enc_sd)[:8])
print("DEC keys sample:", list(dec_sd)[:8])
