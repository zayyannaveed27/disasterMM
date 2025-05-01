import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ───────────── configuration ────────────────────────────────────────────────
npz_path    = Path("/data/wildfirets/new/compressed/359_2020-07-29_2020-08-27.npz")
wanted_date = "2020-08-05"       # or set date_idx = 0‑based integer below
date_idx    = None
percentile_lo, percentile_hi = 2, 98          # contrast‑stretch for continuous vars
save_png    = "new_panel_2020-08-05.png"                            # e.g. Path("panel_2020‑08‑05.png")

# Figure 1: Dataset channels example for 2020-08-05 at (35.5° N, -106.5° W) 
# Figure 1: Dataset channels example for 2020-01-10 at (35.5° N, -106.5° W) 

# ────────────────────────────────────────────────────────────────────────────

cols = ['I1','M4','M3','M11','I2','I3','aerosol depth','NDVI_last','EVI2_last',
        'total precipitation','wind speed','wind direction','minimum temperature',
        'maximum temperature','energy release component','specific humidity',
        'pdsi','LC_Type1','forecast total precipitation','forecast wind speed',
        'forecast wind direction','forecast temperature','forecast specific humidity',
        'slope','aspect','elevation']

# sensible cmaps for each single‑band variable ------------------------------
# ── replace the previous `cmaps = { ... }` block with this one ─────────────
cmaps = {
    # visible / SWIR imagery
    'M11' : 'inferno', 'I2' : 'inferno', 'I3' : 'inferno',

    # vegetation‑health indices
    'NDVI_last' : 'RdYlGn', 'EVI2_last' : 'RdYlGn',

    # precip / wind‑speed / humidity  → sequential Y‑G‑B ramp
    'total precipitation'           : 'viridis',
    'forecast total precipitation'  : 'viridis',
    'wind speed'                    : 'viridis',
    'forecast wind speed'           : 'viridis',
    'specific humidity'             : 'viridis',
    'forecast specific humidity'    : 'viridis',

    # temperatures stay the same
    'minimum temperature'           : 'coolwarm',
    'maximum temperature'           : 'coolwarm',
    'forecast temperature'          : 'coolwarm',

    # other scalar fields
    'aerosol depth'                 : 'cividis',
    'energy release component'      : 'YlOrRd',
    'pdsi'                          : 'BrBG',
    'slope'                         : 'gist_earth',
    'elevation'                     : 'terrain',
    'LC_Type1'                      : 'tab20',   # categorical classes

    # **directional** variables (0–360°) → cyclic twilight
    'wind direction'                : 'twilight',
    'forecast wind direction'       : 'twilight',
    'aspect'                        : 'twilight',
}
# ────────────────────────────────────────────────────────────────────────────

# ---------------------------------------------------------------------------

# load the data --------------------------------------------------------------
with np.load(npz_path, allow_pickle=True) as npz:
    dates = npz['dates']     # ('YYYY‑MM‑DD', …)
    data  = npz['data']      # (T, 26, 224, 224)

if date_idx is None:
    try:
        date_idx = list(dates).index(wanted_date)
    except ValueError:
        raise ValueError(f"{wanted_date} not in file.  Available dates:\n{dates}")

slice_ = data[date_idx]      # (26, 224, 224)

# build RGB composite (I1 / M4 / M3) ----------------------------------------
def stretch(img, lo=2, hi=98):
    """robust percentile stretch to [0,1]"""
    vmin, vmax = np.percentile(img, [lo, hi])
    img = np.clip(img, vmin, vmax)
    return (img - vmin) / (vmax - vmin + 1e-9)

rgb = np.dstack([stretch(slice_[0], percentile_lo, percentile_hi),
                 stretch(slice_[1], percentile_lo, percentile_hi),
                 stretch(slice_[2], percentile_lo, percentile_hi)])

# prepare single‑band images list -------------------------------------------
remaining_names  = cols[3:]                    # 23 labels
remaining_arrays = slice_[3:]                  # (23, 224, 224)

# create figure --------------------------------------------------------------
fig = plt.figure(figsize=(16, 12))
gs  = fig.add_gridspec(5, 6, wspace=0.15, hspace=0.25)

# (0,0) — RGB composite
ax0 = fig.add_subplot(gs[0, 0])
ax0.imshow(rgb)
ax0.set_title("I1 / M4 / M3 (RGB)", fontsize=10)
ax0.axis('off')

# plot the 23 bands ----------------------------------------------------------
plot_idx = 1
for i, (name, band) in enumerate(zip(remaining_names, remaining_arrays)):
    # if 'forecast' in name:
    #     continue
    r, c = divmod(plot_idx, 6)     # +1 shifts by the composite already placed
    ax   = fig.add_subplot(gs[r, c])

    if name == 'LC_Type1':    # categorical — no stretchi`ng
        ax.imshow(band, cmap=cmaps.get(name, 'viridis'), interpolation='nearest')
    else:                     # continuous — stretch
        ax.imshow(stretch(band, percentile_lo, percentile_hi),
                   cmap=cmaps.get(name, 'viridis'))
    ax.set_title(name, fontsize=8)
    ax.axis('off')
    plot_idx += 1

# tidy up & save -------------------------------------------------------------
# fig.suptitle(f"{dates[date_idx]} – channel overview", fontsize=14)
if save_png:
    fig.savefig(save_png, dpi=300, bbox_inches='tight')
plt.show()