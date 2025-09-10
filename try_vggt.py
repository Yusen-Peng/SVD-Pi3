import torch
import torch, numpy as np, matplotlib.pyplot as plt
from matplotlib import cm
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images

def _to_numpy(x):
    return x.detach().float().cpu().numpy()

def _depth_to_color(depth_2d, robust=True):
    # depth_2d: HxW (float). Map to 0..1 for colormap
    d = depth_2d.copy()
    d[np.isinf(d)] = np.nan
    if robust:
        lo, hi = np.nanpercentile(d, [2, 98])  # robust normalization
    else:
        lo, hi = np.nanmin(d), np.nanmax(d)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = 0.0, 1.0
    d = np.clip((d - lo) / (hi - lo + 1e-8), 0, 1)
    return cm.magma(d)[..., :3]  # RGB

def visualize_vggt_predictions(predictions, show_tracks=True, point_stride=16):
    """
    predictions: dict from VGGT
      expects keys: images [B,S,3,H,W], depth [B,S,H,W,1], depth_conf [B,S,H,W],
                    world_points [B,S,H,W,3], (optional) track [B,S,N,2], vis/conf [B,S,N]
    """
    assert "images" in predictions, "Call model in eval mode to get 'images' in predictions."
    imgs = _to_numpy(predictions["images"])            # [B,S,3,H,W]
    depth = _to_numpy(predictions.get("depth"))        # [B,S,H,W,1] or None
    dconf = _to_numpy(predictions.get("depth_conf"))   # [B,S,H,W] or None
    wpts = _to_numpy(predictions.get("world_points"))  # [B,S,H,W,3] or None

    has_depth = depth is not None
    has_conf  = dconf is not None and has_depth
    has_pts   = wpts is not None
    has_trk   = show_tracks and ("track" in predictions)

    if has_trk:
        tracks = _to_numpy(predictions["track"])       # [B,S,N,2] in pixel coords
        vis    = _to_numpy(predictions.get("vis"))     # [B,S,N]
        conf   = _to_numpy(predictions.get("conf"))    # [B,S,N]
    else:
        tracks = vis = conf = None

    B, S = imgs.shape[:2]
    for b in range(B):
        for s in range(S):
            img = np.transpose(imgs[b, s], (1, 2, 0))  # HWC, in [0,1]
            H, W = img.shape[:2]

            # --- Figure layout: image + depth ---
            ncols = 2 if has_depth else 1
            fig = plt.figure(figsize=(6*ncols, 6))
            ax1 = fig.add_subplot(1, ncols, 1)
            ax1.set_title(f"Original Image")
            ax1.imshow(img)
            ax1.axis("off")

            if has_depth:
                d = depth[b, s, ..., 0]  # HxW
                d_rgb = _depth_to_color(d, robust=True)

                if has_conf:
                    # map conf to alpha (0..1). You can tweak scaling.
                    alpha = np.clip(dconf[b, s], 0, 1)
                    # overlay depth on the original image
                    composite = (1 - alpha[..., None]) * img + alpha[..., None] * d_rgb
                    ax2 = fig.add_subplot(1, ncols, 2)
                    ax2.set_title("Depth (color) x Conf (alpha)")
                    ax2.imshow(composite)
                    ax2.axis("off")
                else:
                    ax2 = fig.add_subplot(1, ncols, 2)
                    ax2.set_title("Depth (colorized)")
                    ax2.imshow(d_rgb)
                    ax2.axis("off")

            if has_trk:
                ax_img = ax1  # draw tracks on the image panel
                xs = tracks[b, :, :, 0]  # [S,N]
                ys = tracks[b, :, :, 1]
                # for visibility we can filter by vis/conf if provided
                visible = None
                if vis is not None:
                    visible = vis[b] > 0.5
                elif conf is not None:
                    # conf ∈ [0,1], threshold at 0.3
                    visible = conf[b] > 0.3

                # draw trajectories up to frame s
                T = s + 1
                N = xs.shape[1]
                for n in range(N):
                    if visible is not None and not np.any(visible[:T, n]):
                        continue
                    # polyline up to current frame
                    ax_img.plot(xs[:T, n], ys[:T, n], lw=1.5, alpha=0.9)
                    # current position as dot
                    ax_img.scatter(xs[s, n], ys[s, n], s=12)

            plt.tight_layout()
            plt.show()
            plt.savefig(f"toy_output/depth_b{b}_s{s}.png")

            if has_pts:
                # Subsample grid for speed/readability
                yy = np.arange(0, H, point_stride)
                xx = np.arange(0, W, point_stride)
                grid_y, grid_x = np.meshgrid(yy, xx, indexing="ij")
                pts = wpts[b, s, grid_y, grid_x, :]      # [h', w', 3]
                pts = pts.reshape(-1, 3)
                # Drop NaNs/Infs
                mask = np.isfinite(pts).all(axis=1)
                pts = pts[mask]

                if pts.shape[0] > 0:
                    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
                    fig = plt.figure(figsize=(7, 6))
                    ax3d = fig.add_subplot(111, projection='3d')
                    ax3d.scatter(pts[:,0], pts[:,1], pts[:,2], s=1)
                    ax3d.set_title(f"World points (subsampled) b={b}, s={s}")
                    ax3d.set_xlabel("X"); ax3d.set_ylabel("Y"); ax3d.set_zlabel("Z")
                    # equal-ish aspect
                    rng = np.nanmax(pts, axis=0) - np.nanmin(pts, axis=0)
                    mid = (np.nanmax(pts, axis=0) + np.nanmin(pts, axis=0))/2
                    r = np.max(rng) / 2 if np.isfinite(rng).all() else 1.0
                    ax3d.set_xlim(mid[0]-r, mid[0]+r)
                    ax3d.set_ylim(mid[1]-r, mid[1]+r)
                    ax3d.set_zlim(mid[2]-r, mid[2]+r)
                    plt.tight_layout()
                    plt.show()
                    plt.savefig(f"toy_output/points_b{b}_s{s}.png")

def main():

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # bfloat16 is supported on Ampere GPUs (Compute Capability 8.0+) 
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

    # Initialize the model and load the pretrained weights.
    # This will automatically download the model weights the first time it's run, which may take a while.
    model = VGGT.from_pretrained("facebook/VGGT-1B").to(device)

    # Load and preprocess example images (replace with your own image paths)
    image_names = ["try/image_1.png", "try/image_2.png", "try/image_3.png"]  
    images = load_and_preprocess_images(image_names).to(device)

    model.eval()
    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=dtype):
            # Predict attributes including cameras, depth maps, and point maps.
            predictions = model(images)
    visualize_vggt_predictions(predictions, show_tracks=True, point_stride=16)

if __name__ == "__main__":
    main()
