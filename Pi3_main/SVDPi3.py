#coding:utf8
from typing import OrderedDict
import warnings
warnings.filterwarnings("ignore", message=".*RoPE2D.*")
warnings.filterwarnings("ignore", message=".*version instead.*")
warnings.filterwarnings("ignore", message=".*_register_pytree_node is deprecated.*")
import os
import sys

current_path = os.path.dirname(os.path.abspath(__file__))
parent_path  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_path not in sys.path:
    sys.path.insert(0, parent_path)

import pandas as pd
from contextlib import nullcontext
import argparse
import plotly.express as px
from safetensors.torch import load_file
from tqdm import tqdm
import torch
import torch.nn as nn
import csv
from pi3.utils.basic import load_images_as_tensor, write_ply
from pi3.utils.geometry import depth_edge
from pi3.models.pi3 import Pi3
from pi3.models.layers.block import BlockRope
from SVD_LLM.utils.data_utils import *
from SVD_LLM.utils.peft import PeftModel
from SVD_LLM.component.svd_llama import SVD_LlamaAttention, SVD_LlamaMLP
from SVD_LLM.component.svd_mistral import SVD_MistralAttention, SVD_MistralMLP
from SVD_LLM.component.svd_opt import SVDOPTDecoderLayer
from SVD_LLM.utils.model_utils import *
from SVD_LLM.evaluater import *

from fvcore.nn import FlopCountAnalysis

import torch.profiler as prof

def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def simple_efficiency(model, imgs, dtype):
    with torch.no_grad():
        with torch.amp.autocast('cuda', dtype=dtype):
            sync()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            res = model(imgs[None])
            end.record()
            torch.cuda.synchronize()
            model_ms = start.elapsed_time(end)  # milliseconds
    print(f"✅Model forward: {model_ms:.2f} ms")
    fps = (imgs.shape[0] / (model_ms / 1000.0))
    print(f"✅Throughput: {fps:.2f} frames/sec")
    return res

def profiler_efficiency(model, imgs, dtype, csv_path="profile.csv"):
    with prof.profile(
        activities=[prof.ProfilerActivity.CPU, prof.ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
        with_stack=False
    ) as p:
        with torch.no_grad(), torch.amp.autocast('cuda', dtype=dtype):
            res = model(imgs[None])

    # Collect key averages
    events = p.key_averages()

    # Write to CSV
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        # header
        writer.writerow([
            "Name",
            "CPU time total (us)",
            "CUDA time total (us)",
            "Calls",
            "Input shapes",
            "Self CPU Mem (KB)",
            "Self CUDA Mem (KB)"
        ])
        # rows
        for evt in events:
            writer.writerow([
                evt.key,
                evt.cpu_time_total,
                evt.cuda_time_total,
                evt.count,
                evt.input_shapes,
                evt.self_cpu_memory_usage / 1024,
                evt.self_cuda_memory_usage / 1024
            ])

    print(f"✅Profiler results written to {csv_path}")
    return res

def build_profiler_plots(
    csv_path: str = "profile.csv",
    html_path: str = "topk_cuda_ops.html",
    png_path: str = "topk_cuda_ops.png",
    top_k: int = 10,
):
    """
    Build an interactive horizontal bar chart of the Top-K ops by CUDA time total.
    Saves an interactive HTML and (if kaleido is available) a static PNG.

    Parameters
    ----------
    csv_path : str
        Path to the CSV produced by `profiler_efficiency`.
    html_path : str
        Output path for the interactive HTML.
    png_path : str
        Output path for the static PNG; requires `kaleido` installed.
    top_k : int
        Number of top ops to display, ranked by CUDA time total (ms).

    Returns
    -------
    pd.DataFrame
        The top-K dataframe used for plotting (sorted by CUDA time).
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    # Read and coerce numeric columns
    df = pd.read_csv(csv_path)

    # Standardize column names in case of minor variations
    col_map = {
        "Name": "Name",
        "CPU time total (us)": "CPU_us",
        "CUDA time total (us)": "CUDA_us",
        "Calls": "Calls",
        "Input shapes": "Input shapes",
        "Self CPU Mem (KB)": "Self CPU Mem (KB)",
        "Self CUDA Mem (KB)": "Self CUDA Mem (KB)",
    }
    # Ensure all expected columns exist
    for k in col_map:
        if k not in df.columns:
            raise ValueError(f"Expected column '{k}' not found in {csv_path}")

    df = df.rename(columns=col_map)

    # Coerce numerics
    for c in ["CPU_us", "CUDA_us", "Calls", "Self CPU Mem (KB)", "Self CUDA Mem (KB)"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

    # Aggregate by op Name (multiple entries can occur across shapes/threads)
    agg = (
        df.groupby("Name", as_index=False)
          .agg({
              "CPU_us": "sum",
              "CUDA_us": "sum",
              "Calls": "sum",
              "Self CPU Mem (KB)": "sum",
              "Self CUDA Mem (KB)": "sum",
              "Input shapes": lambda s: "; ".join(sorted(map(str, set(s))))
          })
    )

    # Derived metrics (ms & percentages)
    agg["CPU_ms"] = agg["CPU_us"] / 1000.0
    agg["CUDA_ms"] = agg["CUDA_us"] / 1000.0

    total_cuda_ms = agg["CUDA_ms"].sum()
    if total_cuda_ms <= 0:
        # Avoid division by zero
        total_cuda_ms = 1e-9

    agg["CUDA_%"] = 100.0 * agg["CUDA_ms"] / total_cuda_ms

    # Rank by CUDA time and keep Top-K
    top = (
        agg.sort_values("CUDA_ms", ascending=False)
           .head(max(1, int(top_k)))
           .copy()
    )

    # Nice labels
    top["Label"] = top["Name"]

    # Build figure
    title = f"Top {len(top)} Ops by CUDA Time (total={total_cuda_ms:.1f} ms)"
    height = 50 * len(top) + 220  # scale height with K (roomy)
    fig = px.bar(
        top.sort_values("CUDA_ms", ascending=True),
        x="CUDA_ms",
        y="Label",
        orientation="h",
        text=top["CUDA_%"].map(lambda x: f"{x:.1f}%"),
        hover_data={
            "CUDA_ms": ":.3f",
            "CPU_ms": ":.3f",
            "CUDA_%": ":.2f",
            "Calls": True,
            "Self CPU Mem (KB)": True,
            "Self CUDA Mem (KB)": True,
            "Input shapes": True,
            "Label": False,
        },
        title=title,
        labels={
            "CUDA_ms": "CUDA time total (ms)",
            "Label": "Op name",
            "CPU_ms": "CPU time total (ms)",
            "CUDA_%": "Share of total CUDA time",
        },
    )

    # Layout tweaks for long labels
    fig.update_layout(
        height=height,
        margin=dict(l=380, r=80, t=80, b=60),  # big left margin for long op names
        xaxis=dict(title="CUDA time total (ms)"),
        yaxis=dict(automargin=True),
        uniformtext_minsize=10,
        font=dict(size=12),
    )
    fig.update_traces(textposition="outside", cliponaxis=False)

    # Save interactive HTML
    fig.write_html(html_path, include_plotlyjs="cdn", full_html=True)

    # Try to save PNG (requires kaleido)
    try:
        fig.write_image(png_path, scale=2)  # needs `pip install -U kaleido`
    except Exception as e:
        print(f"[build_profiler_plots] PNG export skipped (install 'kaleido' to enable). Reason: {e}")

    print(f"[build_profiler_plots] Wrote HTML to: {html_path}")
    if os.path.exists(png_path):
        print(f"[build_profiler_plots] Wrote PNG  to: {png_path}")

    return top


@torch.no_grad()
def profle_svdllm(name, model, calib_loader, dev):
    if "llama" in name or "mistral" in name or "vicuna" in name:
        layers = model.model.layers
    elif "opt" in name:
        layers = model.model.decoder.layers
    model = model.to(dev)
    print("Start obtaining the whitening matrix...")
    def hook(module, input, output):
        inp = input[0].detach().float()
        if inp.dim() == 2:   # for opt
            inp = inp.unsqueeze(0)
        adds = torch.matmul(inp.transpose(1,2), inp)
        adds_sum = torch.sum(adds, dim=0)
        module.raw_scaling_diag_matrix += adds_sum
        del inp, adds, adds_sum
        torch.cuda.empty_cache()
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            module.raw_scaling_diag_matrix = 0
            module.register_forward_hook(hook)
    for batch in tqdm(calib_loader):
        batch = {k: v.to(dev) for k, v in batch.items()}
        model(**batch)
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            module._forward_hooks.clear()
    torch.cuda.empty_cache()
    model = model.cpu()
    for i in range(len(layers)):
        subset = find_layers(layers[i])
        for name in subset:
            subset[name].raw_scaling_diag_matrix = subset[name].raw_scaling_diag_matrix.cpu()
    profiling_mat = {}
    print("Start Cholesky Decomposition...")
    for i in tqdm(range(len(layers))):
        layer_profile = {}
        subset = find_layers(layers[i])
        for name in subset:
            raw_scaling_diag_matrix = subset[name].raw_scaling_diag_matrix.double().to(dev)
            try:
                scaling_diag_matrix = torch.linalg.cholesky(raw_scaling_diag_matrix)
            except Exception as e:
                print("Warning: eigen scaling_diag_matrix is not positive!")
                eigenvalues = torch.linalg.eigvalsh(raw_scaling_diag_matrix)
                raw_scaling_diag_matrix += (- eigenvalues[0] + 1e-6) * torch.eye(raw_scaling_diag_matrix.shape[0]).to(dev)
                scaling_diag_matrix = torch.linalg.cholesky(raw_scaling_diag_matrix)
                eigenvalues = None
                del eigenvalues
            layer_profile[name] = scaling_diag_matrix.cpu()
            scaling_diag_matrix = raw_scaling_diag_matrix = subset[name].raw_scaling_diag_matrix = None
            del scaling_diag_matrix, raw_scaling_diag_matrix, subset[name].raw_scaling_diag_matrix
            torch.cuda.empty_cache()
        profiling_mat[i] = layer_profile
    return profiling_mat

@torch.no_grad()
def profle_svdllm_low_resource(model_name, model, calib_loader, dev):
    if "opt" in model_name:
        layers = model.model.decoder.layers
        model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.to(dev)
        model.model.decoder.final_layer_norm = model.model.decoder.final_layer_norm.to(dev)
        model.model.decoder.embed_positions = model.model.decoder.embed_positions.to(dev)
    else:
        layers = model.model.layers
        model.model.embed_tokens = model.model.embed_tokens.to(dev)
        model.model.norm = model.model.norm.to(dev)
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (len(calib_loader), model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {'i': 0, 'attention_mask': None, "position_ids": None}
    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp.cpu()
            cache['i'] += 1
            if cache['attention_mask'] is None:
                cache['attention_mask'] = kwargs['attention_mask'].cpu()
                if "opt" not in model_name:
                    cache['position_ids'] = kwargs['position_ids'].cpu()
            else:
                cache['attention_mask'] = torch.cat((cache['attention_mask'], kwargs['attention_mask'].cpu()), dim=0)
                if "opt" not in model_name:
                    cache['position_ids'] = torch.cat((cache['position_ids'], kwargs['position_ids'].cpu()), dim=0)
            raise ValueError
    layers[0] = Catcher(layers[0])
    for batch in calib_loader:
        try:
            batch = {k: v.to(dev) for k, v in batch.items()}
            model(**batch)
        except ValueError:
            pass
    layers[0] = layers[0].module
    layers[0] = layers[0].cpu()
    if "opt" in model_name:
        model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.cpu()
        model.model.decoder.final_layer_norm = model.model.decoder.final_layer_norm.cpu()
        model.model.decoder.embed_positions = model.model.decoder.embed_positions.cpu()
    else:  
        model.model.embed_tokens = model.model.embed_tokens.cpu()
        model.model.norm = model.model.norm.cpu()
    torch.cuda.empty_cache()
    outs = torch.zeros_like(inps)
    attention_masks = cache['attention_mask']
    if "opt" not in model_name:
        position_ids = cache['position_ids']
    profiling_mat = {}
    for i in tqdm(range(len(layers))):
        layer_profile = {}
        layer = layers[i].to(dev)
        subset = find_layers(layer)        
        def hook(module, input, output):
            inp = input[0].detach().float()
            if inp.dim() == 2:  # for opt
                inp = inp.unsqueeze(0)
            adds = torch.matmul(inp.transpose(1,2), inp)
            adds_sum = torch.sum(adds, dim=0)
            module.scaling_diag_matrix += adds_sum
            del inp, adds, adds_sum, output
            torch.cuda.empty_cache()
        handles = []
        for name in subset:
            subset[name].scaling_diag_matrix = 0
            handles.append(subset[name].register_forward_hook(hook))
        for j in range(inps.shape[0]):
            if "opt" not in model_name:
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_masks[j].unsqueeze(0).to(dev), position_ids=position_ids[j].unsqueeze(0).to(dev))[0]
            else:
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_masks[j].unsqueeze(0).to(dev))[0]
        for h in handles:
            h.remove()
        layer = layer.cpu()
        for name in subset:
            subset[name].scaling_diag_matrix = subset[name].scaling_diag_matrix.cpu()
        torch.cuda.empty_cache()
        for name in subset:
            raw_scaling_diag_matrix = subset[name].scaling_diag_matrix.double().to(dev)
            try:
                scaling_diag_matrix = torch.linalg.cholesky(raw_scaling_diag_matrix)
            except Exception as e:
                print("Warning: eigen scaling_diag_matrix is not positive!")
                eigenvalues = torch.linalg.eigvalsh(raw_scaling_diag_matrix)
                raw_scaling_diag_matrix += (- eigenvalues[0] + 1e-6) * torch.eye(raw_scaling_diag_matrix.shape[0]).to(dev)
                scaling_diag_matrix = torch.linalg.cholesky(raw_scaling_diag_matrix)
                eigenvalues = None
                del eigenvalues
            layer_profile[name] = scaling_diag_matrix.cpu()
            scaling_diag_matrix = raw_scaling_diag_matrix = subset[name].raw_scaling_diag_matrix = None
            del scaling_diag_matrix, raw_scaling_diag_matrix, subset[name].raw_scaling_diag_matrix
            torch.cuda.empty_cache()
        layers[i] = layer.cpu()
        profiling_mat[i] = layer_profile
        inps = outs
        torch.cuda.empty_cache()
    return profiling_mat


@torch.no_grad()
def Pi3_profile_svdllm_low_resource(
    model: nn.Module,
    calib_batches: List[Dict[str, torch.Tensor]],
    device: torch.device,
    autocast: bool = True,
    dtype: torch.dtype = torch.float16,
    eps: float = 1e-6,
) -> Dict[str, torch.Tensor]:
    """
    Stream calibration data through Pi3 and compute per-module whitening factors.

    Returns
    -------
    profiling_mat: Dict[str, Tensor]
        Maps module_key -> Cholesky factor L of covariance (so that Cov ≈ L @ L^T).
    """
    model.eval().to(device)

    # choose targets (attention linear layers and MLP linear layers)
    targets = OrderedDict()
    # Pi3.decoder is nn.ModuleList[BlockRope]
    for i, blk in enumerate(model.decoder):
        blk: BlockRope = blk # type check
        # attention
        if hasattr(blk, "attn"):
            attn = blk.attn
            if hasattr(attn, "qkv") and isinstance(attn.qkv, nn.Linear):
                targets[f"decoder.{i}.attn.qkv"] = attn.qkv
            if hasattr(attn, "proj") and isinstance(attn.proj, nn.Linear):
                targets[f"decoder.{i}.attn.proj"] = attn.proj
        # mlp (ffn)
        if hasattr(blk, "mlp"):
            mlp = blk.mlp
            if hasattr(mlp, "fc1") and isinstance(mlp.fc1, nn.Linear):
                targets[f"decoder.{i}.mlp.fc1"] = mlp.fc1
            if hasattr(mlp, "fc2") and isinstance(mlp.fc2, nn.Linear):
                targets[f"decoder.{i}.mlp.fc2"] = mlp.fc2
    
    print(f"✅Found {len(targets)} Linear targets in Pi3.decoder to whiten")

    # initialize per-module accumulators
    # G = X^T * X, N = total rows collected
    grams: Dict[str, torch.Tensor] = {}
    counts: Dict[str, int] = {}
    for k, lin in targets.items():
        in_dim = lin.in_features
        grams[k] = torch.zeros((in_dim, in_dim), dtype=torch.float64, device=device)
        counts[k] = 0

    # define hooks to collect statistics during the calibration forward passes
    handles = []
    def make_pre_hook(key: str):
        def _hook(module: nn.Linear, inp):
            # inp is a tuple; grab first
            x = inp[0]
            # expected shapes: (..., in_features)
            x = x.detach()
            # collapse all leading dims to rows
            x = x.reshape(-1, x.shape[-1]).to(device, dtype=torch.float32)
            # center? (optional) — for calibration whitening, uncentered is okay
            G = x.T @ x   # (in_features, in_features) in float32
            grams[key] += G.to(torch.float64)
            counts[key] += x.shape[0]
        return _hook

    for key, lin in targets.items():
        lin: nn.Linear = lin  # type check
        handles.append(lin.register_forward_pre_hook(make_pre_hook(key)))
    print(f"✅Registered {len(handles)} forward hooks!")


    # run forward streaming through batches
    amp_ctx = (
        torch.amp.autocast(device_type="cuda", dtype=dtype) if (autocast and device.type == "cuda")
        else nullcontext()
    )

    for b in tqdm(calib_batches):
        imgs = b["pixel_values"].to(device)
        # Pi3.forward expects (B, N, C, H, W). Our sampler returns (B, C, H, W).
        # Use N=1.
        imgs = imgs.unsqueeze(1)

        with amp_ctx:
            # here, PyTorch automatically invokes every registered forward_pre_hook
            _ = model(imgs)
    print(f"✅Completed streaming {len(calib_batches)} calibration batches")

    # remove hooks
    for h in handles:
        h.remove()
    print(f"✅Removed all {len(handles)} forward hooks")
    torch.cuda.synchronize(device) if device.type == "cuda" else None

    # build whitening matrices
    profiling_mat: Dict[str, torch.Tensor] = {}

    num_modules = len(targets)
    print(f"Building {num_modules} Cholesky factors (on CPU)...")
    fail_case = 0

    for key in targets.keys():
        n = max(1, counts[key])

        # 1) CPU float64 & symmetrize
        cov = (grams[key] / n).to(torch.float64).cpu()
        cov = 0.5 * (cov + cov.T)

        d = cov.shape[0]
        I = torch.eye(d, dtype=cov.dtype, device=cov.device)

        # scale-aware base shrinkage (Ledoit-Wolf style tiny alpha)
        mu = float(cov.trace() / max(1, d))
        base_eps = 1e-6 * max(1.0, mu)   # adapt to magnitude
        cov_j = cov + base_eps * I

        # 2) try Cholesky on CPU
        try:
            L = torch.linalg.cholesky(cov_j)
        except Exception:
            fail_case += 1
            evals, Q = torch.linalg.eigh(cov)  # CPU, float64, symmetric
            lam = torch.clamp(evals, min=base_eps)
            L = Q @ torch.diag(torch.sqrt(lam))

        profiling_mat[key] = L  # keep on CPU to save VRAM

    print(f"✅{num_modules - fail_case}/{num_modules} succeeded with Cholesky, {fail_case}/{num_modules} used EVD fallback")

    # clean up
    for k in grams:
        grams[k] = None
    torch.cuda.empty_cache()

    return profiling_mat

 
@torch.no_grad()
def whitening(model_name, model, profiling_mat, ratio, dev):
    model.eval()
    if 'opt' in model_name:
        layers = model.model.decoder.layers
    else:
        layers = model.model.layers
    print("Start SVD decomposition after whitening...")
    for i in tqdm(range(len(layers))):
        layer = layers[i]
        subset = find_layers(layer)
        #### Replace Attn, MLP ####
        if "llama" in model_name or "vicuna" in model_name:
            svd_attn = SVD_LlamaAttention(config=model.config, ratio=ratio)
            svd_mlp = SVD_LlamaMLP(hidden_size=layer.hidden_size, intermediate_size=model.config.intermediate_size, hidden_act=model.config.hidden_act, ratio=ratio)
        elif "mistral" in model_name:
            svd_attn = SVD_MistralAttention(config=model.config, ratio=ratio)
            svd_mlp = SVD_MistralMLP(config=model.config, ratio=ratio)
        elif 'opt' in model_name:
            svd_decoder = SVDOPTDecoderLayer(model.config, ratio=ratio)
        #### Replace Attn, MLP ####
        for name in subset:
            W = subset[name].weight.data.float().to(dev)
            dtype = W.dtype
            scaling_diag_matrix = profiling_mat[i][name].to(dev)
            try:
                scaling_matrix_inv = torch.linalg.inv(scaling_diag_matrix)
            except Exception as e:
                print("Warning: scaling_diag_matrix is not full rank!")
                scaling_diag_matrix += 1e-6 * torch.eye(scaling_diag_matrix.shape[0]).to(dev)
                scaling_matrix_inv = torch.linalg.inv(scaling_diag_matrix)
            scaling_diag_matrix = scaling_diag_matrix.float()
            scaling_matrix_inv = scaling_matrix_inv.float()
            W_scale = torch.matmul(W, scaling_diag_matrix)
            U, S, VT = torch.linalg.svd(W_scale, full_matrices=False)
            num_s_after_trunc = int(W.shape[0] * W.shape[1] * ratio / (W.shape[0] + W.shape[1]))
            truc_s = S[:num_s_after_trunc]
            truc_u = U[:, :num_s_after_trunc]
            truc_v = torch.matmul(VT[:num_s_after_trunc, :], scaling_matrix_inv)
            truc_sigma = torch.diag(truc_s)
            #### Replace Attn, MLP ####
            sqrtSigma = torch.sqrt(truc_sigma)
            svd_u = torch.matmul(truc_u, sqrtSigma).cpu().to(dtype)
            svd_v = torch.matmul(sqrtSigma, truc_v).cpu().to(dtype)
            if 'opt' in model_name:
                if "q_proj" in name:
                    svd_decoder.self_attn.q_u_proj.weight.data = svd_u
                    svd_decoder.self_attn.q_v_proj.weight.data = svd_v
                    svd_decoder.self_attn.q_u_proj.bias.data = layer.self_attn.q_proj.bias.data  # the linear layer in OPT has bias, which is different from LLaMA and Mistral
                elif "k_proj" in name:
                    svd_decoder.self_attn.k_u_proj.weight.data = svd_u
                    svd_decoder.self_attn.k_v_proj.weight.data = svd_v
                    svd_decoder.self_attn.k_u_proj.bias.data = layer.self_attn.k_proj.bias.data
                elif "v_proj" in name:
                    svd_decoder.self_attn.v_u_proj.weight.data = svd_u
                    svd_decoder.self_attn.v_v_proj.weight.data = svd_v
                    svd_decoder.self_attn.v_u_proj.bias.data = layer.self_attn.v_proj.bias.data
                elif "out_proj" in name:
                    svd_decoder.self_attn.out_u_proj.weight.data = svd_u
                    svd_decoder.self_attn.out_v_proj.weight.data = svd_v
                    svd_decoder.self_attn.out_u_proj.bias.data = layer.self_attn.out_proj.bias.data
                elif "fc1" in name:
                    svd_decoder.fc1_u_proj.weight.data = svd_u
                    svd_decoder.fc1_v_proj.weight.data = svd_v
                    svd_decoder.fc1_u_proj.bias.data = layer.fc1.bias.data
                elif "fc2" in name:
                    svd_decoder.fc2_u_proj.weight.data = svd_u
                    svd_decoder.fc2_v_proj.weight.data = svd_v
                    svd_decoder.fc2_u_proj.bias.data = layer.fc2.bias.data
                    svd_decoder.self_attn_layer_norm = layer.self_attn_layer_norm
                    svd_decoder.final_layer_norm = layer.final_layer_norm
                    layers[i] = svd_decoder
            else:
                if "q_proj" in name:
                    svd_attn.q_u_proj.weight.data = svd_u
                    svd_attn.q_v_proj.weight.data = svd_v
                elif "k_proj" in name:
                    svd_attn.k_u_proj.weight.data = svd_u
                    svd_attn.k_v_proj.weight.data = svd_v
                elif "v_proj" in name:
                    svd_attn.v_u_proj.weight.data = svd_u
                    svd_attn.v_v_proj.weight.data = svd_v
                elif "o_proj" in name:
                    svd_attn.o_u_proj.weight.data = svd_u
                    svd_attn.o_v_proj.weight.data = svd_v
                    layer.self_attn =  svd_attn
                elif "gate_proj" in name:
                    svd_mlp.gate_u_proj.weight.data = svd_u
                    svd_mlp.gate_v_proj.weight.data = svd_v
                elif "down_proj" in name:
                    svd_mlp.down_u_proj.weight.data = svd_u
                    svd_mlp.down_v_proj.weight.data = svd_v
                elif "up_proj" in name:
                    svd_mlp.up_u_proj.weight.data = svd_u
                    svd_mlp.up_v_proj.weight.data = svd_v
                    layer.mlp = svd_mlp
            W = W_scale = scaling_matrix_inv = scaling_diag_matrix = U = S = VT  = truc_s = truc_u = truc_v = sqrtSigma = None
            del  W, W_scale, scaling_matrix_inv, scaling_diag_matrix, U, S, VT, truc_s, truc_u, truc_v, sqrtSigma
        del layer
        torch.cuda.empty_cache()


def Pi3_whitening(model_name, model, profiling_mat, ratio, dev):
    raise NotImplementedError()


@torch.no_grad()
def whitening_local_update(model_name, model, dataloader, profiling_mat, ratio, dev, direct_update=False):
    print("Start SVD decomposition then update...")
    use_cache = model.config.use_cache
    model.config.use_cache = False
    if "opt" in model_name:
        layers = model.model.decoder.layers
        model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.to(dev)
        model.model.decoder.final_layer_norm = model.model.decoder.final_layer_norm.to(dev)
        model.model.decoder.embed_positions = model.model.decoder.embed_positions.to(dev)
    else:
        layers = model.model.layers
        model.model.embed_tokens = model.model.embed_tokens.to(dev)
        model.model.norm = model.model.norm.to(dev)
    model.model.norm = model.model.norm.to(dev)
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (len(dataloader), model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {'i': 0, 'attention_mask': None, "position_ids": None}
    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            if cache['attention_mask'] is None:
                cache['attention_mask'] = kwargs['attention_mask']
                if "opt" not in model_name:
                    cache['position_ids'] = kwargs['position_ids']
            else:
                cache['attention_mask'] = torch.cat((cache['attention_mask'], kwargs['attention_mask']), dim=0)
                if "opt" not in model_name:
                    cache['position_ids'] = torch.cat((cache['position_ids'], kwargs['position_ids']), dim=0)
            raise ValueError
    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(dev))
        except ValueError:
            pass
    layers[0] = layers[0].module
    layers[0] = layers[0].cpu()
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    model.model.norm = model.model.norm.cpu()
    torch.cuda.empty_cache()
    outs = torch.zeros_like(inps)
    attention_masks = cache['attention_mask']
    if "opt" not in model_name:
        position_ids = cache['position_ids']
    for i in tqdm(range(len(layers))):
        layer = layers[i].to(dev)
        subset = find_layers(layer)
        gpts = {}
        if "llama" in model_name or "vicuna" in model_name:
            svd_attn = SVD_LlamaAttention(config=model.config, ratio=ratio)
            svd_mlp = SVD_LlamaMLP(hidden_size=layer.hidden_size, intermediate_size=model.config.intermediate_size, hidden_act=model.config.hidden_act, ratio=ratio)
        elif "mistral" in model_name:
            svd_attn = SVD_MistralAttention(config=model.config, ratio=ratio)
            svd_mlp = SVD_MistralMLP(config=model.config, ratio=ratio)
        elif 'opt' in model_name:
            svd_decoder = SVDOPTDecoderLayer(model.config, ratio=ratio)
        for name in subset:
            if profiling_mat is not None:
                scaling_diag_matrix = profiling_mat[i][name].to(dev)
            else: 
                scaling_diag_matrix = None
            gpts[name] = local_update(subset[name], scaling_diag_matrix = scaling_diag_matrix, ratio=ratio, name=name, direct_update=direct_update)
        
        def add_batch(name):
            def tmp(_, inp, out):
                gpts[name].add_batch_update_u(inp[0].data, out.data)
            return tmp
        handles = []
        for name in gpts:
            handles.append(subset[name].register_forward_hook(add_batch(name)))
        if "opt" not in model_name:
            outs = layer(inps, attention_mask=attention_masks, position_ids=position_ids)[0]
        else:
            outs = layer(inps, attention_mask=attention_masks)[0]
        for h in handles:
            h.remove()
        for name in gpts:
            svd_u, svd_v = gpts[name].fasterprune()
            svd_u, svd_v = svd_u.to(dtype), svd_v.to(dtype)
            if 'opt' in model_name:
                if "q_proj" in name:
                    svd_decoder.self_attn.q_u_proj.weight.data = svd_u
                    svd_decoder.self_attn.q_v_proj.weight.data = svd_v
                    svd_decoder.self_attn.q_u_proj.bias.data = layer.self_attn.q_proj.bias.data  # the linear layer in OPT has bias, which is different from LLaMA and Mistral
                elif "k_proj" in name:
                    svd_decoder.self_attn.k_u_proj.weight.data = svd_u
                    svd_decoder.self_attn.k_v_proj.weight.data = svd_v
                    svd_decoder.self_attn.k_u_proj.bias.data = layer.self_attn.k_proj.bias.data
                elif "v_proj" in name:
                    svd_decoder.self_attn.v_u_proj.weight.data = svd_u
                    svd_decoder.self_attn.v_v_proj.weight.data = svd_v
                    svd_decoder.self_attn.v_u_proj.bias.data = layer.self_attn.v_proj.bias.data
                elif "out_proj" in name:
                    svd_decoder.self_attn.out_u_proj.weight.data = svd_u
                    svd_decoder.self_attn.out_v_proj.weight.data = svd_v
                    svd_decoder.self_attn.out_u_proj.bias.data = layer.self_attn.out_proj.bias.data
                elif "fc1" in name:
                    svd_decoder.fc1_u_proj.weight.data = svd_u
                    svd_decoder.fc1_v_proj.weight.data = svd_v
                    svd_decoder.fc1_u_proj.bias.data = layer.fc1.bias.data
                elif "fc2" in name:
                    svd_decoder.fc2_u_proj.weight.data = svd_u
                    svd_decoder.fc2_v_proj.weight.data = svd_v
                    svd_decoder.fc2_u_proj.bias.data = layer.fc2.bias.data
                    svd_decoder.self_attn_layer_norm = layer.self_attn_layer_norm
                    svd_decoder.final_layer_norm = layer.final_layer_norm
                    layers[i] = svd_decoder
            else:
                if "q_proj" in name:
                    svd_attn.q_u_proj.weight.data = svd_u
                    svd_attn.q_v_proj.weight.data = svd_v
                elif "k_proj" in name:
                    svd_attn.k_u_proj.weight.data = svd_u
                    svd_attn.k_v_proj.weight.data = svd_v
                elif "v_proj" in name:
                    svd_attn.v_u_proj.weight.data = svd_u
                    svd_attn.v_v_proj.weight.data = svd_v
                elif "o_proj" in name:
                    svd_attn.o_u_proj.weight.data = svd_u
                    svd_attn.o_v_proj.weight.data = svd_v
                    layer.self_attn =  svd_attn
                elif "gate_proj" in name:
                    svd_mlp.gate_u_proj.weight.data = svd_u
                    svd_mlp.gate_v_proj.weight.data = svd_v
                elif "down_proj" in name:
                    svd_mlp.down_u_proj.weight.data = svd_u
                    svd_mlp.down_v_proj.weight.data = svd_v
                elif "up_proj" in name:
                    svd_mlp.up_u_proj.weight.data = svd_u
                    svd_mlp.up_v_proj.weight.data = svd_v
                    layer.mlp = svd_mlp
        layer = layer.to(dev)
        if "opt" not in model_name:
            outs = layer(inps, attention_mask=attention_masks, position_ids=position_ids)[0]
        else:
            outs = layer(inps, attention_mask=attention_masks)[0]
        layers[i] = layer.cpu()
        del gpts
        torch.cuda.empty_cache()
        inps = outs
        outs = None
        del outs
    model.config.use_cache = use_cache


class local_update:
    def __init__(self, layer, scaling_diag_matrix, ratio, name, direct_update=False):
        self.layer = layer
        self.name = name
        self.dev = self.layer.weight.device
        # W = layer.weight.data.clone()
        W = layer.weight.data.clone()
        self.rows = W.shape[0]
        self.columns = W.shape[1]
        if direct_update:
            self.U, self.S, self.VT = torch.linalg.svd(W.data, full_matrices=False)
        else: 
            try:
                scaling_matrix_inv = torch.linalg.inv(scaling_diag_matrix)
            except Exception as e:
                print("Warning: scaling_diag_matrix is not full rank!")
                scaling_diag_matrix += 1e-6 * torch.eye(scaling_diag_matrix.shape[0])
                scaling_matrix_inv = torch.linalg.inv(scaling_diag_matrix)
            scaling_diag_matrix = scaling_diag_matrix.float()
            scaling_matrix_inv = scaling_matrix_inv.float()
            W_scale = torch.matmul(W, scaling_diag_matrix)
            self.U, self.S, self.VT = torch.linalg.svd(W_scale, full_matrices=False)  
        # trucation SVD
        num_s_after_trunc = int(W.shape[0] * W.shape[1] * ratio / (W.shape[0] + W.shape[1]))
        self.truc_s = self.S[:num_s_after_trunc].cuda()
        self.truc_u = self.U[:, :num_s_after_trunc].cuda()
        if direct_update:
            self.truc_v = self.VT[:num_s_after_trunc, :].cuda()
        else:
            self.truc_v = torch.matmul(self.VT[:num_s_after_trunc, :].cuda(), scaling_matrix_inv)
        self.truc_sigma = torch.diag(self.truc_s)
        self.new_w = torch.matmul(self.truc_u, torch.matmul(self.truc_sigma, self.truc_v[:num_s_after_trunc, :]))
        # intialize H for close form solution
        self.updated_err = self.error = 0

    def add_batch_update_u(self, inp, out):
        inps = inp.view(inp.shape[0] * inp.shape[1], inp.shape[2])
        outs = out.view(out.shape[0] * out.shape[1], out.shape[2])
        new_w = torch.matmul(self.truc_u, torch.matmul(self.truc_sigma, self.truc_v))
        new_output = inps.matmul(new_w.t())
        self.error = torch.sqrt(torch.sum((outs - new_output)**2)).item() / torch.norm(outs, p='fro').item()
        # print(f"truncted error: {self.error}")
        x =  torch.matmul(torch.matmul(inps, self.truc_v.T), self.truc_sigma)
        self.updated_uT = torch.linalg.lstsq(x,outs).solution
        updated_output = torch.matmul(torch.matmul(torch.matmul(inps, self.truc_v.T), self.truc_sigma), self.updated_uT)
        self.updated_error = torch.sqrt(torch.sum((outs - updated_output)**2)).item() / torch.norm(outs, p='fro').item()
        # print(f"updated error: {self.updated_error}")
        inps = outs = new_output = updated_output = x = new_w = None
        del inps, outs, new_output, updated_output, x, new_w
        torch.cuda.empty_cache()
        # print(f"Finish {self.name}"
    
    def fasterprune(self):
        sqrtSigma = torch.sqrt(self.truc_sigma)
        self.appendU = self.updated_uT.t().matmul(sqrtSigma)
        self.appendV = sqrtSigma.matmul(self.truc_v)
        return self.appendU, self.appendV


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument('--model', type=str, default='jeffwan/llama-7b-hf', help='LLaMA model to load, pass `jeffwan/llama-7b-hf`')
    parser.add_argument('--model_path', type=str, default=None, help='local compressed model path or whitening information path')
    parser.add_argument('--run_low_resource', action='store_true', help='whether to run whitening in low resource, exp, compress LLaMA-7B below 15G gpu')
    parser.add_argument('--dataset', type=str, default='wikitext2',help='Where to extract calibration data from [wikitext2, ptb, c4]')
    parser.add_argument('--whitening_nsamples', type=int, default=512, help='Number of calibration data samples for whitening.')
    parser.add_argument('--updating_nsamples', type=int, default=16, help='Number of calibration data samples for udpating.')
    parser.add_argument('--save_path', type=str, default=None, help='the path to save the compressed model checkpoints.`')
    parser.add_argument('--seed',type=int, default=0, help='Seed for sampling the calibration data')
    parser.add_argument('--DEV', type=str, default="cuda", help='device')
    parser.add_argument('--model_seq_len', type=int, default=2048, help='the default sequence length of the LLM')
    parser.add_argument('--eval_batch_size', type=int, default=4, help='inference bactch size')
    parser.add_argument('--gen_seq_len', type=int, default=1024, help='generated sequence len for efficiency evaluation')
    parser.add_argument('--step', type=int, default=4, help='the step to run the compression')
    parser.add_argument('--lora', type=str, default=None, help='the lora updated weight path to run the accuracy evaluation')
    parser.add_argument("--interval", type=int, default=-1, help="Interval to sample image. Default: 1 for images dir, 10 for video")
    parser.add_argument("--ckpt", type=str, default=None, help="Path to the model checkpoint file. Default: None")
    parser.add_argument('--ratio', type=float, default=0.2, help='Target compression ratio,(0,1), default=0.2, means only keeping about 20% of the params.')
    parser.add_argument("--device", type=str, default='cuda', help="Device to run inference on ('cuda' or 'cpu'). Default: 'cuda'")
    parser.add_argument("--calibration_dataset_path", type=str, default="/data/wanghaoxuan/sintel", help="Path to the calibration dataset.")

    args = parser.parse_args()
    args.ratio = 1- args.ratio


    if args.step == 1:
        """
            Whitening only (no updates)
        """

        print(f'Whitening only (no updates) with sampling interval: {args.interval}')
        device = torch.device(args.device)
        if args.ckpt is not None:
            model = Pi3().to(device).eval()
            if args.ckpt.endswith('.safetensors'): 
                weight = load_file(args.ckpt)
            else:
                weight = torch.load(args.ckpt, map_location=device, weights_only=False)
            
            model.load_state_dict(weight)
        else:
            model = Pi3.from_pretrained("yyfz233/Pi3").to(device).eval()

        model = model.eval()
        print("✅ model loaded.")

        # collect calibration data
        print("Start collecting calibration data...")
        cali_white_data = Pi3_get_calib_train_data(
            root=args.calibration_dataset_path,
            nsamples=args.whitening_nsamples
        )
        print(f"✅ collected {len(cali_white_data)} calibration batches (~{sum(b['pixel_values'].shape[0] for b in cali_white_data)} images).")

        # derive the whitening matrix via profiling
        profiling_mat = Pi3_profile_svdllm_low_resource(model, cali_white_data, device, autocast=True, dtype=torch.float16, eps=1e-6)




        # # TODO: apply whitening
        # Pi3_whitening(args.model, model, profiling_mat, args.ratio, args.DEV)

        # # save the model
        # torch.save({'model': model}, args.save_path + "/" + args.model.replace("/", "_").replace("-", "_") +'_whitening_only_' + str(args.ratio) + '.pt')
    
    
    
    elif args.step == 2:
        """
            Whitening + local update (light finetune)
        """

        model, tokenizer = get_model_from_huggingface(model_id=args.model)
        dataloader, _ = get_loaders(args.dataset, nsamples=args.updating_nsamples, seed=args.seed, tokenizer=tokenizer, seqlen=args.model_seq_len)
        model = model.eval()
        model = model.float()  # need to set to float
        cali_white_data = get_calib_train_data(args.dataset, tokenizer, args.whitening_nsamples, seqlen=args.model_seq_len)
        profiling_mat = profle_svdllm_low_resource(args.model, model, cali_white_data, args.DEV)
        if args.save_path is not None:
            torch.save(profiling_mat, args.save_path + "/" + args.model.replace("/", "_").replace("-", "_") + '_profiling_'+ args.dataset + '_' + str(args.whitening_nsamples)  + '_' + str(args.seed)+ '.pt')
        
        whitening_local_update(args.model, model, dataloader, profiling_mat, args.ratio, args.DEV)
        if args.save_path is not None:
            torch.save({'model': model, 'tokenizer': tokenizer}, args.save_path + "/" + args.model.replace("/", "_").replace("-", "_") +'_whitening_then_update_' + str(args.ratio) + '.pt')  # fp32
    
    
    
    elif args.step == 3:
        """
            Local update only (no whitening)
        """


        model, tokenizer = get_model_from_huggingface(args.model)
        model = model.eval()
        model = model.float()
        dataloader, _ = get_loaders(args.dataset, nsamples=args.updating_nsamples, seed=args.seed, tokenizer=tokenizer, seqlen=args.model_seq_len)
        whitening_local_update(model_name=args.model, model=model, dataloader=dataloader, profiling_mat=None, ratio=args.ratio, dev=args.DEV, direct_update=True)
        if args.save_path is not None:
            torch.save({'model': model, 'tokenizer': tokenizer}, args.save_path + "/" + args.model.replace("/", "_").replace("-", "_") +'_update_only_' + str(args.ratio) + '.pt')   # fp32
    
    
    
    
    elif args.step >= 4:
        """
            Evaluation modes.
        """

        print(f"evaluating {args.model_path}...")
        if args.model_path == "original":
            model, tokenizer = get_model_from_huggingface(args.model)
        else:
            model, tokenizer = get_model_from_local(args.model_path)
            if args.lora is not None:
                model = PeftModel.from_pretrained(
                    model,
                    args.lora,
                    torch_dtype=torch.float16,
                )
                model = model.merge_and_unload()
                torch.save({'model': model, 'tokenizer': tokenizer}, args.lora + '/merge.pt')
        model.eval()
        model = model.float()
        model = model.to(args.DEV)




        if args.step == 4:
            """
                Accuracy evaluation
            """


            ppl_eval(model, tokenizer, datasets=['wikitext2'], model_seq_len=args.model_seq_len, batch_size=args.eval_batch_size, device=args.DEV)
        
        
        elif args.step == 5:
            """
                Efficiency evaluation
            """

            eff_eval(model, tokenizer, generated_len=args.gen_seq_len, batch_size=args.eval_batch_size, device=args.DEV)
    
    print("✅ALL DONE!")


def pi3_tutorial():
    parser = argparse.ArgumentParser(description="Run inference with the Pi3 model.")
    
    parser.add_argument("--data_path", type=str, default='examples/skating.mp4',
                        help="Path to the input image directory or a video file.")
    parser.add_argument("--save_path", type=str, default='examples/result.ply',
                        help="Path to save the output .ply file.")
    parser.add_argument("--interval", type=int, default=-1,
                        help="Interval to sample image. Default: 1 for images dir, 10 for video")
    parser.add_argument("--ckpt", type=str, default=None,
                        help="Path to the model checkpoint file. Default: None")
    parser.add_argument("--device", type=str, default='cuda',
                        help="Device to run inference on ('cuda' or 'cpu'). Default: 'cuda'")
    parser.add_argument("--efficiency_measure", type=str, default='simple', choices=['simple', 'profiler'],
                        help="Type of efficiency measurement to perform. Default: 'simple'")

    args = parser.parse_args()
    if args.interval < 0:
        args.interval = 10 if args.data_path.endswith('.mp4') else 1
    print(f'Sampling interval: {args.interval}')


    # 1. Prepare model
    print(f"Loading model...")
    device = torch.device(args.device)
    if args.ckpt is not None:
        model = Pi3().to(device).eval()
        if args.ckpt.endswith('.safetensors'):
            from safetensors.torch import load_file
            weight = load_file(args.ckpt)
        else:
            weight = torch.load(args.ckpt, map_location=device, weights_only=False)
        
        model.load_state_dict(weight)
    else:
        model = Pi3.from_pretrained("yyfz233/Pi3").to(device).eval()
        # or download checkpoints from `https://huggingface.co/yyfz233/Pi3/resolve/main/model.safetensors`, and `--ckpt ckpts/model.safetensors`

    # 2. Prepare input data
    # The load_images_as_tensor function will print the loading path
    imgs = load_images_as_tensor(args.data_path, interval=args.interval).to(device) # (N, 3, H, W)

    # 3. Infer
    print("Running model inference...")
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16


    if args.efficiency_measure == 'profiler':
        top_k = 10
        res = profiler_efficiency(model, imgs, dtype)
        build_profiler_plots(
            csv_path="profile.csv",
            top_k=top_k
        )
    elif args.efficiency_measure == 'simple':
        res = simple_efficiency(model, imgs, dtype)
    else:
        raise ValueError(f"Unknown efficiency_measure: {args.efficiency_measure}")

    # 4. process mask
    masks = torch.sigmoid(res['conf'][..., 0]) > 0.1
    non_edge = ~depth_edge(res['local_points'][..., 2], rtol=0.03)
    masks = torch.logical_and(masks, non_edge)[0]

    # 5. Save points
    print(f"Saving point cloud to: {args.save_path}")
    write_ply(res['points'][0][masks].cpu(), imgs.permute(0, 2, 3, 1)[masks], args.save_path)
    print("Done.")


if __name__ == "__main__":

    print("✅Hello world")
    main()