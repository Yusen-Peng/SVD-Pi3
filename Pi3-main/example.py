import torch
import argparse
import time
import csv
import os
import pandas as pd
import plotly.express as px
from fvcore.nn import FlopCountAnalysis

import torch.profiler as prof
from pi3.utils.basic import load_images_as_tensor, write_ply
from pi3.utils.geometry import depth_edge
from pi3.models.pi3 import Pi3


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





if __name__ == '__main__':
    # --- Argument Parsing ---
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

    # from pi3.utils.debug import setup_debug
    # setup_debug()

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