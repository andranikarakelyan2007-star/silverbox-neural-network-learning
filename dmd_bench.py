"""dmd_bench.py

Simulated bench: laser -> DMD (amplitude only) -> scattering medium T -> camera.
    Eout = |T Ein|^2   (+ shot noise and read noise)

The search (focusing) follows the lecture algorithm:
  * the DMD is split into macropixels, start from a random binary pattern;
  * after each "transmission" read the intensity in the focus region and in the proximal background;
  * the background is split into several regions (sectors of a ring around the focus);
    enhancement = mean focus intensity / mean of the background region means;
  * flip some macropixels, keep the flip if the enhancement grew, repeat.

Public entry point for the agent: run_focusing(...)  -> JSON-serialisable dict.
"""

import json
import time
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Circle  # noqa: E402

DMD_N = 64          # DMD is DMD_N x DMD_N micromirrors
CAM_N = 48          # camera is CAM_N x CAM_N pixels
BENCH_SEED = 1      # the scattering medium is fixed for all experiments
MEAN_COUNTS = 100.0  # mean speckle intensity (camera counts) for a random 50% pattern
READ_NOISE = 3.0     # camera read noise, counts (std)
SATURATION = 65535.0
VALID_MACRO = [m for m in (1, 2, 4, 8, 16, 32) if DMD_N % m == 0]


class Bench:
    """Fixed scattering medium. T maps every DMD micromirror to every camera pixel."""

    def __init__(self, seed: int = BENCH_SEED, dmd_n: int = DMD_N, cam_n: int = CAM_N):
        rng = np.random.default_rng(seed)
        self.dmd_n, self.cam_n = dmd_n, cam_n
        shape = (cam_n * cam_n, dmd_n * dmd_n)
        re = rng.standard_normal(shape, dtype=np.float32)
        im = rng.standard_normal(shape, dtype=np.float32)
        self.T = ((re + 1j * im) / np.sqrt(2.0)).astype(np.complex64)   # circular complex Gaussian
        # a random 50% pattern gives <|E|^2> = 0.5 * dmd_n^2 -> scale it to MEAN_COUNTS
        self.gain = MEAN_COUNTS / (0.5 * dmd_n * dmd_n)
        self._macro = {}

    def macro_T(self, m: int) -> np.ndarray:
        """Transmission matrix of macropixels: columns of one m x m block are summed."""
        if m not in self._macro:
            g = self.dmd_n // m
            t = self.T.reshape(self.cam_n * self.cam_n, g, m, g, m).sum(axis=(2, 4))
            self._macro[m] = np.ascontiguousarray(t.reshape(self.cam_n * self.cam_n, g * g))
        return self._macro[m]

    def camera(self, field: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Intensity |E|^2 (in counts) with shot (Poisson) noise and Gaussian read noise."""
        inten = self.gain * (np.abs(field) ** 2).astype(np.float64)
        shot = rng.poisson(inten).astype(np.float64)
        out = shot + rng.normal(0.0, READ_NOISE, size=inten.shape)
        return np.clip(out, 0.0, SATURATION)


_BENCH = None


def get_bench() -> Bench:
    global _BENCH
    if _BENCH is None:
        _BENCH = Bench()
    return _BENCH


def make_masks(cam_n, focus_radius, bg_gap, bg_width, n_bg):
    yy, xx = np.mgrid[:cam_n, :cam_n]
    c = cam_n // 2
    r = np.hypot(yy - c, xx - c)
    outer = focus_radius + bg_gap + bg_width
    if outer > c - 1:
        raise ValueError(
            f"background reaches radius {outer:.1f} px but the camera allows at most {c - 1} px: "
            "reduce focus_radius + bg_gap + bg_width"
        )
    focus = r <= focus_radius
    ring = (r > focus_radius + bg_gap) & (r <= outer)
    ang = np.arctan2(yy - c, xx - c) + np.pi
    sector = np.minimum((ang / (2 * np.pi) * n_bg).astype(int), n_bg - 1)
    bg = [ring & (sector == k) for k in range(n_bg)]
    if focus.sum() == 0 or any(b.sum() == 0 for b in bg):
        raise ValueError("a region is empty: increase the radii/width or use fewer background regions")
    return focus, bg, r


def enhancement_from(img_roi_focus, img_roi_bg):
    """focus mean / mean of the background-region means."""
    return float(np.mean(img_roi_focus) / np.mean([np.mean(b) for b in img_roi_bg]))


def _draw_regions(ax, cam_n, focus_radius, bg_gap, bg_width):
    c = cam_n // 2
    ax.add_patch(Circle((c, c), focus_radius + 0.5, fill=False, ec="red", lw=1.2))
    ax.add_patch(Circle((c, c), focus_radius + bg_gap + 0.5, fill=False, ec="cyan", lw=0.8, ls="--"))
    ax.add_patch(Circle((c, c), focus_radius + bg_gap + bg_width + 0.5, fill=False, ec="cyan", lw=0.8, ls="--"))


def run_focusing(
    macropixel_size: int = 8,
    focus_radius: float = 2.0,
    bg_gap: float = 2.0,
    bg_width: float = 8.0,
    n_bg_regions: int = 4,
    n_iterations: int = 3000,
    flips_per_iteration: int = 1,
    seed: int = 0,
    out_dir: str = "results",
    run_name: str = "run",
) -> dict:
    """Run one focusing experiment on the simulated bench and return the enhancement + image paths."""
    t_start = time.time()
    bench = get_bench()
    m = int(macropixel_size)
    if m not in VALID_MACRO:
        raise ValueError(f"macropixel_size must be one of {VALID_MACRO}")
    g = bench.dmd_n // m
    n_mp = g * g
    n_bg = int(n_bg_regions)
    n_it = int(n_iterations)
    k = int(flips_per_iteration)
    if not (1 <= n_bg <= 12):
        raise ValueError("n_bg_regions must be 1..12")
    if not (1 <= n_it <= 200000):
        raise ValueError("n_iterations must be 1..200000")
    if not (1 <= k <= max(1, n_mp // 2)):
        raise ValueError(f"flips_per_iteration must be 1..{max(1, n_mp // 2)} for {n_mp} macropixels")
    focus_radius, bg_gap, bg_width = float(focus_radius), float(bg_gap), float(bg_width)
    if focus_radius < 0 or bg_gap < 0 or bg_width <= 0:
        raise ValueError("focus_radius >= 0, bg_gap >= 0, bg_width > 0 are required")

    focus_m, bg_m, _ = make_masks(bench.cam_n, focus_radius, bg_gap, bg_width, n_bg)
    flat_f = focus_m.ravel()
    flat_bg = [b.ravel() for b in bg_m]
    roi_mask = flat_f.copy()
    for b in flat_bg:
        roi_mask |= b
    roi = np.flatnonzero(roi_mask)
    # positions of the regions inside the ROI vector
    pos = -np.ones(roi_mask.size, dtype=int)
    pos[roi] = np.arange(roi.size)
    f_idx = pos[np.flatnonzero(flat_f)]
    bg_idx = [pos[np.flatnonzero(b)] for b in flat_bg]

    def eta(img_roi):
        return enhancement_from(img_roi[f_idx], [img_roi[i] for i in bg_idx])

    Tm = bench.macro_T(m)
    T_roi = np.ascontiguousarray(Tm[roi])           # only the pixels we read during the search
    rng = np.random.default_rng(seed)

    a = (rng.random(n_mp) < 0.5).astype(np.float32)  # random binary start pattern (amplitude 0/1)
    a0 = a.copy()
    E = T_roi @ a

    hist_it, hist_eta = [], []
    log_every = max(1, n_it // 200)
    accepted = 0
    for it in range(1, n_it + 1):
        idx = rng.choice(n_mp, size=k, replace=False)
        delta = (1.0 - 2.0 * a[idx]).astype(np.float32)          # 0->1 : +1,  1->0 : -1
        E_new = E + T_roi[:, idx] @ delta
        eta_cur = eta(bench.camera(E, rng))                       # paired measurements, fresh noise each
        eta_new = eta(bench.camera(E_new, rng))
        if eta_new > eta_cur:
            a[idx] = 1.0 - a[idx]
            E = E_new
            accepted += 1
        if it % 500 == 0:
            E = T_roi @ a                                         # kill float32 drift
        if it % log_every == 0 or it == n_it:
            hist_it.append(it)
            hist_eta.append(eta_cur)

    # ---- final measurements with fresh noise: average over several camera shots ----
    n_shots = 5
    full_rng = np.random.default_rng(seed + 12345)

    def measure(pattern):
        field = Tm @ pattern
        imgs = [bench.camera(field, full_rng) for _ in range(n_shots)]
        etas = [eta(im.ravel()[roi]) for im in imgs]
        return imgs[0].reshape(bench.cam_n, bench.cam_n), float(np.mean(etas)), float(np.std(etas))

    img0, eta0, eta0_sd = measure(a0)
    img1, eta1, eta1_sd = measure(a)
    f_int = float(np.mean(img1.ravel()[flat_f]))
    b_int = float(np.mean([np.mean(img1.ravel()[b]) for b in flat_bg]))

    # ---- figure: initial image, final image, convergence ----
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    png = out / f"{run_name}.png"
    vmax = float(np.percentile(img1, 99.9))
    fig, axs = plt.subplots(1, 4, figsize=(17, 4))
    axs[0].imshow(a0.reshape(g, g), cmap="gray", vmin=0, vmax=1)
    axs[0].set_title(f"DMD start pattern ({g}x{g} macropixels)")
    axs[1].imshow(img0, cmap="inferno", vmin=0, vmax=vmax)
    _draw_regions(axs[1], bench.cam_n, focus_radius, bg_gap, bg_width)
    axs[1].set_title(f"camera, random pattern\nenhancement {eta0:.2f}")
    axs[2].imshow(img1, cmap="inferno", vmin=0, vmax=vmax)
    _draw_regions(axs[2], bench.cam_n, focus_radius, bg_gap, bg_width)
    axs[2].set_title(f"camera, optimised pattern\nenhancement {eta1:.2f}")
    axs[3].plot(hist_it, hist_eta)
    axs[3].set_xlabel("iteration")
    axs[3].set_ylabel("measured enhancement")
    axs[3].set_title("convergence")
    axs[3].grid(alpha=0.3)
    fig.suptitle(
        f"{run_name}: macropixel {m}px, focus r={focus_radius}, bg gap={bg_gap}, width={bg_width}, "
        f"{n_bg} regions, {n_it} iter, {k} flip/iter", fontsize=10)
    fig.tight_layout()
    fig.savefig(png, dpi=110)
    plt.close(fig)
    np.save(out / f"{run_name}_camera_final.npy", img1)
    np.save(out / f"{run_name}_camera_start.npy", img0)

    return {
        "enhancement": round(eta1, 3),
        "enhancement_std_over_5_shots": round(eta1_sd, 3),
        "enhancement_initial_random_pattern": round(eta0, 3),
        "focus_intensity_counts": round(f_int, 1),
        "background_intensity_counts": round(b_int, 1),
        "n_macropixels": n_mp,
        "accepted_flips": accepted,
        "iterations": n_it,
        "focus_pixels": int(flat_f.sum()),
        "background_pixels": int(sum(b.sum() for b in flat_bg)),
        "runtime_s": round(time.time() - t_start, 2),
        "image_path": str(png),
    }


if __name__ == "__main__":
    res = run_focusing()
    print(json.dumps(res, indent=2))
