#!/usr/bin/env python3
import sys
import json
from pycore.tikzeng import to_head, to_cor, to_begin, to_input, to_end, to_connection
from pycore.blocks import (
    to_ConvConvRelu,
    to_Pool,
    block_2ConvPool,
    block_Unconv,
    to_skip,
    to_ConvSoftMax,
)

def make_plot_arch(plan_path):
    # 1) load plan.json
    with open(plan_path) as f:
        plans = json.load(f)
    cfg = plans["3d_fullres"]["architecture"]["arch_kwargs"]

    n_enc = cfg["n_stages"]
    feats = cfg["features_per_stage"]
    n_conv = cfg["n_conv_per_stage"]
    strides = cfg["strides"]
    n_dec = cfg["n_conv_per_stage_decoder"]

    arch = [
        to_head('..'),
        to_cor(),
        to_begin(),
        # input placeholder (patch size)
        to_input(f'{plans["3d_fullres"]["patch_size"]}'),  
    ]

    # --- Encoder ---
    for i in range(n_enc):
        name = f"ccr_b{i+1}"
        s_f = feats[i]
        n_f = (s_f, s_f)
        # size heuristics: larger features -> smaller display
        size = (max(2, 6 - i), max(2, 6 - i), max(2, 6 - i))
        off = f"({i*2},0,0)"
        to = f"({i*2},0,0)"
        # conv block
        arch.append(
            to_ConvConvRelu(
                name=name, 
                s_filer=feats[0],           # arbitrary front-end fmap
                n_filer=n_f, 
                offset=off, 
                to=to, 
                width=size[:2], 
                height=size[0]*10, 
                depth=size[1]*10
            )
        )
        # pooling except last encoder stage
        if i < n_enc - 1:
            arch.append(
                to_Pool(
                    name=f"pool_b{i+1}", 
                    offset=off, 
                    to=f"({name}-east)", 
                    width=1, 
                    height=size[0]*10 - 4, 
                    depth=size[1]*10 - 4, 
                    opacity=0.5
                )
            )

    # --- Bottleneck ---
    i = n_enc
    arch.append(
        to_ConvConvRelu(
            name=f"ccr_b{i+1}",
            s_filer=feats[0],
            n_filer=(feats[-1], feats[-1]),
            offset=f"({i*2},0,0)",
            to=f"(pool_b{n_enc}-east)",
            width=(8,8),
            height=8,
            depth=8,
            caption="Bottleneck"
        )
    )
    arch.append(to_connection(f"pool_b{n_enc}", f"ccr_b{n_enc+1}"))

    # --- Decoder ---
    for j in range(n_enc-1):
        stage = n_enc + 1 + j
        bot  = f"ccr_b{stage}" if j == 0 else f"end_b{stage}"
        top  = f"end_b{stage+1}"
        n_f  = feats[-(j+2)]
        size = (max(2, 6 - (n_enc-1-j)),)*2 + (5,)
        arch.extend(
            block_Unconv(
                name=f"b{stage+1}",
                botton=bot,
                top=top,
                s_filer=feats[-1],
                n_filer=n_f,
                offset=f"({stage*2 + 0.1},0,0)",
                size=size,
                opacity=0.5
            )
        )
        # skip connection from corresponding encoder ccr_b
        enc_idx = n_enc - 1 - j
        arch.append(to_skip(of=f"ccr_b{enc_idx+1}", to=f"ccr_res_b{stage+1}", pos=1.25))

    # softmax & end
    arch.append(
        to_ConvSoftMax(
            name="soft1",
            s_filer=feats[0],
            offset=f"({(n_enc*2)+0.75},0,0)",
            to=f"(end_b{n_enc+1}-east)",
            width=1,
            height=feats[0]*10,
            depth=feats[0]*10,
            caption="Softmax"
        )
    )
    arch.append(to_connection(f"end_b{n_enc+1}", "soft1"))
    arch.append(to_end())

    return arch

def main():
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <plan.json>")
        sys.exit(1)
    plan_fn = sys.argv[1]
    arch = make_plot_arch(plan_fn)
    out_tex = plan_fn.replace('.json', '.tex')
    from pycore.tikzeng import to_generate
    to_generate(arch, out_tex)
    print(f"Generated {out_tex}")

if __name__ == "__main__":
    main()
