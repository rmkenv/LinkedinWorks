
"""s


# 🏗️ Parcel-Level Construction Change Detection
### Sentinel-2 Optical + Sentinel-1 RTC SAR — No GEE Required
**Data source:** Microsoft Planetary Computer (free, no account needed)  
**Access method:** STAC → `stackstac` → `xarray` (cloud-native COG streaming)  
**SAR:** Pre-processed S1 RTC (radiometrically terrain corrected) — no SNAP needed

```
Pipeline
────────
STAC search (PC)
  ├── Sentinel-2 L2A COGs  → cloud mask → spectral indices (NDVI, NDBI, BSI, EVI)
  └── Sentinel-1 RTC COGs  → speckle filter → VV/VH log-ratio change
                                    ↓
                         Fuse into change stack
                                    ↓
                   Weighted Construction Change Index (CCI)
                                    ↓
                    Zonal stats per parcel → GeoJSON + maps
```

## 0. Install Dependencies
"""

!pip install stackstac pystac-client planetary-computer \
             rioxarray rasterio geopandas rasterstats \
             xarray dask folium matplotlib seaborn \
             scikit-learn shapely pyproj numba --quiet

import numpy as np
import pandas as pd
import xarray as xr
import rioxarray
import geopandas as gpd
import json
import warnings
warnings.filterwarnings('ignore')

import stackstac
import pystac_client
import planetary_computer as pc

import rasterio
from rasterio.enums import Resampling
from rasterstats import zonal_stats

import folium
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from IPython.display import display

print('All imports successful.')

"""## 1. Configuration"""

# ─────────────────────────────────────────
# USER CONFIGURATION
# ─────────────────────────────────────────

# Date ranges
PRE_START  = '2015-01-01'
PRE_END    = '2020-06-30'
POST_START = '2025-01-01'
POST_END   = '2026-03-31'

# Bounding box [lon_min, lat_min, lon_max, lat_max]
# Default: small area in Baltimore, MD — swap in your AOI
BBOX = [-76.68364421069992, 39.24233810620235, -76.68164421069991, 39.24433810620235]

# Max cloud cover % for Sentinel-2
CLOUD_THRESHOLD = 20

# Output resolution (metres) — 10 or 20
RESOLUTION = 10

# Parcel input: 'manual' (grid from bbox) | 'geojson' | 'shapefile'
PARCEL_INPUT = 'manual'
PARCEL_PATH  = '/content/parcels.geojson'

# Construction Change Index threshold to flag a parcel (0–1)
CCI_THRESHOLD = 0.45

# Outputs
OUTPUT_GEOJSON = '/content/parcel_cci_results.geojson'
OUTPUT_MAP     = '/content/parcel_cci_map.html'

# Planetary Computer STAC endpoint (no auth needed)
PC_STAC_URL = 'https://planetarycomputer.microsoft.com/api/stac/v1'

print('Configuration loaded.')
print(f'AOI: {BBOX}')
print(f'Pre period : {PRE_START} → {PRE_END}')
print(f'Post period: {POST_START} → {POST_END}')

"""## 2. Load Parcel Geometries"""

from shapely.geometry import box, mapping

def make_synthetic_parcels(bbox, rows=3, cols=3):
    """Subdivide a bounding box into a grid of synthetic parcels."""
    lon_min, lat_min, lon_max, lat_max = bbox
    lon_step = (lon_max - lon_min) / cols
    lat_step = (lat_max - lat_min) / rows
    geoms, ids = [], []
    pid = 0
    for r in range(rows):
        for c in range(cols):
            geoms.append(box(
                lon_min + c * lon_step, lat_min + r * lat_step,
                lon_min + (c+1) * lon_step, lat_min + (r+1) * lat_step
            ))
            ids.append(f'P{pid:03d}')
            pid += 1
    return gpd.GeoDataFrame({'parcel_id': ids}, geometry=geoms, crs='EPSG:4326')


if PARCEL_INPUT == 'geojson':
    parcels_gdf = gpd.read_file(PARCEL_PATH).to_crs('EPSG:4326')
    if 'parcel_id' not in parcels_gdf.columns:
        parcels_gdf['parcel_id'] = [f'P{i:04d}' for i in range(len(parcels_gdf))]
elif PARCEL_INPUT == 'shapefile':
    parcels_gdf = gpd.read_file(PARCEL_PATH).to_crs('EPSG:4326')
    if 'parcel_id' not in parcels_gdf.columns:
        parcels_gdf['parcel_id'] = [f'P{i:04d}' for i in range(len(parcels_gdf))]
else:
    parcels_gdf = make_synthetic_parcels(BBOX)

print(f'Parcels loaded: {len(parcels_gdf)}')
display(parcels_gdf.head())

"""## 3. Connect to Planetary Computer STAC"""

catalog = pystac_client.Client.open(
    PC_STAC_URL,
    modifier=pc.sign_inplace  # signs asset URLs for authenticated COG access
)
print('Connected to Planetary Computer STAC.')

# Helper: search STAC and return signed items
def search_items(collection, bbox, date_range, query=None, limit=100):
    search = catalog.search(
        collections=[collection],
        bbox=bbox,
        datetime=date_range,
        query=query,
        limit=limit
    )
    items = list(search.get_items())
    print(f'  {collection}: {len(items)} items found for {date_range}')
    return items

"""## 4. Sentinel-2 L2A — Load & Cloud-Mask"""

# ── Search S2 L2A ──
S2_COLLECTION = 'sentinel-2-l2a'
S2_BANDS = ['B02', 'B03', 'B04', 'B08', 'B8A', 'B11', 'B12', 'SCL']

print('Searching Sentinel-2 scenes...')
s2_pre_items  = search_items(S2_COLLECTION, BBOX, f'{PRE_START}/{PRE_END}',
                              query={'eo:cloud_cover': {'lt': CLOUD_THRESHOLD}})
s2_post_items = search_items(S2_COLLECTION, BBOX, f'{POST_START}/{POST_END}',
                              query={'eo:cloud_cover': {'lt': CLOUD_THRESHOLD}})

assert len(s2_pre_items) > 0,  'No pre-period S2 scenes found — widen date range or raise cloud threshold'
assert len(s2_post_items) > 0, 'No post-period S2 scenes found — widen date range or raise cloud threshold'

def load_s2_stack(items, bbox, resolution=RESOLUTION):
    stack = stackstac.stack(
        items,
        assets=S2_BANDS,
        resolution=resolution,
        bounds_latlon=bbox,
        epsg=32618,
        resampling=Resampling.bilinear,
        dtype='float64',       # float64 accepts np.nan natively
        fill_value=np.nan,     # safe now — no need for the 0 workaround
        rescale=False,
    ).assign_coords(band=S2_BANDS)  # move this up, drop the .where(stack != 0) line

    scl = stack.sel(band='SCL')
    refl = stack.drop_sel(band='SCL') / 10000.0

    bad_pixels = (scl == 3) | (scl == 8) | (scl == 9) | (scl == 10)
    refl = refl.where(~bad_pixels)

    composite = refl.median(dim='time', skipna=True)
    return composite.compute()

print(f'Pre scenes: {len(s2_pre_items)}  |  Post scenes: {len(s2_post_items)}')

print('Loading S2 post-period composite...')
s2_post = load_s2_stack(s2_post_items, BBOX)
print(f's2_post shape: {s2_post.shape}  dtype: {s2_post.dtype}')

"""## 5. Compute Spectral Indices"""

def spectral_indices(da):
    """
    Compute NDVI, NDBI, BSI, EVI, MNDWI from a band-indexed DataArray.
    Band names must match S2_BANDS (B02, B03, B04, B08, B8A, B11, B12).
    Returns dict of 2-D DataArrays.
    """
    nir  = da.sel(band='B08').astype('float32')
    red  = da.sel(band='B04').astype('float32')
    blue = da.sel(band='B02').astype('float32')
    grn  = da.sel(band='B03').astype('float32')
    sw1  = da.sel(band='B11').astype('float32')

    ndvi  = (nir - red)  / (nir + red  + 1e-9)
    ndbi  = (sw1 - nir)  / (sw1 + nir  + 1e-9)
    bsi   = ((sw1 + red) - (nir + blue)) / ((sw1 + red) + (nir + blue) + 1e-9)
    evi   = 2.5 * (nir - red) / (nir + 6*red - 7.5*blue + 1 + 1e-9)
    mndwi = (grn - sw1)  / (grn + sw1  + 1e-9)

    return {'NDVI': ndvi.clip(-1, 1),
            'NDBI': ndbi.clip(-1, 1),
            'BSI':  bsi.clip(-1, 1),
            'EVI':  evi.clip(-3, 3),
            'MNDWI': mndwi.clip(-1, 1)}

# Fix: Add the missing definition for s2_pre
print('Loading S2 pre-period composite...')
s2_pre = load_s2_stack(s2_pre_items, BBOX)
print(f's2_pre shape: {s2_pre.shape}  dtype: {s2_pre.dtype}')

idx_pre  = spectral_indices(s2_pre)
idx_post = spectral_indices(s2_post)

# Change layers (post − pre)
ndvi_change = (idx_post['NDVI'] - idx_pre['NDVI']).rename('NDVI_change')
ndbi_change = (idx_post['NDBI'] - idx_pre['NDBI']).rename('NDBI_change')
bsi_change  = (idx_post['BSI']  - idx_pre['BSI'] ).rename('BSI_change')
evi_change  = (idx_post['EVI']  - idx_pre['EVI'] ).rename('EVI_change')

print('Spectral indices computed:')
for name, arr in idx_pre.items():
    print(f'  {name}: min={float(arr.min()):.3f}  max={float(arr.max()):.3f}')

"""## 6. Sentinel-1 RTC SAR — Load & Change Detection"""

# Planetary Computer hosts Sentinel-1 RTC (Radiometrically Terrain Corrected)
# VV and VH bands in linear power units — we convert to dB for log-ratio change

S1_COLLECTION = 'sentinel-1-rtc'
S1_BANDS      = ['vv', 'vh']

print('Searching Sentinel-1 RTC scenes...')
s1_pre_items  = search_items(S1_COLLECTION, BBOX, f'{PRE_START}/{PRE_END}')
s1_post_items = search_items(S1_COLLECTION, BBOX, f'{POST_START}/{POST_END}')

assert len(s1_pre_items) > 0,  'No S1 pre scenes found'
assert len(s1_post_items) > 0, 'No S1 post scenes found'

def load_s1_stack(items, bbox, resolution=RESOLUTION):
    """
    Load S1 RTC stack, apply focal mean speckle filter, return dB median composite.
    """
    from scipy.ndimage import uniform_filter

    stack = stackstac.stack(
        items,
        assets=S1_BANDS,
        resolution=resolution,
        bounds_latlon=bbox,
        epsg=32618,
        resampling=Resampling.bilinear,
        dtype='float64', # Changed to float64 for compatibility with np.nan
        fill_value=np.nan,
        rescale=False, # Add rescale=False here
    ).assign_coords(band=S1_BANDS)

    # Convert linear power → dB
    stack_db = 10 * np.log10(stack.where(stack > 0))

    # Median composite across time
    composite = stack_db.median(dim='time', skipna=True).compute()

    # Simple 3×3 spatial mean speckle filter per band
    filtered_bands = []
    for b in S1_BANDS:
        arr = composite.sel(band=b).values
        arr_filt = uniform_filter(np.where(np.isnan(arr), 0, arr), size=3)
        # Restore NaN mask
        arr_filt[np.isnan(arr)] = np.nan
        filtered_bands.append(arr_filt)

    filtered = composite.copy(data=np.stack(filtered_bands, axis=0))
    return filtered


print('Loading S1 pre-period composite...')
s1_pre  = load_s1_stack(s1_pre_items,  BBOX)
print('Loading S1 post-period composite...')
s1_post = load_s1_stack(s1_post_items, BBOX)

# Log-ratio change (dB): positive = backscatter increase = new structures
vv_logratio = (s1_post.sel(band='vv') - s1_pre.sel(band='vv')).rename('VV_logratio')
vh_logratio = (s1_post.sel(band='vh') - s1_pre.sel(band='vh')).rename('VH_logratio')

# VV temporal variance (pre+post combined) → active surface = high variance
print('Computing SAR temporal variance...')
s1_all_items = s1_pre_items + s1_post_items
s1_all = stackstac.stack(
    s1_all_items, assets=['vv'], resolution=RESOLUTION,
    bounds_latlon=BBOX, epsg=32618, dtype='float64', fill_value=np.nan, # Changed to float64 for compatibility with np.nan
    rescale=False # Add rescale=False here too
)
s1_all_db = 10 * np.log10(s1_all.where(s1_all > 0))
vv_variance = s1_all_db.var(dim='time', skipna=True).compute().squeeze().rename('VV_variance')

print('Sentinel-1 SAR change layers computed.')
print(f'  VV log-ratio range: {float(vv_logratio.min()):.2f} → {float(vv_logratio.max()):.2f} dB')

"""## 7. Fuse Layers → Construction Change Index (CCI)"""

def pct_normalize(arr, pct_lo=2, pct_hi=98):
    """Percentile-stretch a numpy array to 0–1."""
    valid = arr[~np.isnan(arr)]
    if len(valid) == 0:
        return arr
    lo = np.percentile(valid, pct_lo)
    hi = np.percentile(valid, pct_hi)
    if hi == lo:
        return np.zeros_like(arr)
    normed = (arr - lo) / (hi - lo)
    return np.clip(normed, 0, 1)


# Construction signals and their weights
# Positive = increasing with construction activity
SIGNALS = {
    'NDVI_loss':    (-ndvi_change.values,  0.15),  # vegetation loss
    'NDBI_gain':    ( ndbi_change.values,  0.20),  # built-up gain
    'BSI_gain':     ( bsi_change.values,   0.25),  # bare soil / excavation
    'EVI_loss':     (-evi_change.values,   0.10),  # canopy loss
    'VV_gain':      ( vv_logratio.values,  0.15),  # SAR backscatter increase
    'VH_gain':      ( vh_logratio.values,  0.05),  # cross-pol change
    'VV_variance':  ( vv_variance.values,  0.10),  # temporal instability
}

# Weighted sum of normalized layers
# Use NDVI as reference for shape; align all arrays to same spatial extent
ref_shape = ndvi_change.values.shape

cci_array = np.zeros(ref_shape, dtype='float32')
for name, (arr, weight) in SIGNALS.items():
    # Resize SAR arrays to match S2 if different (shouldn't be if same EPSG+resolution)
    if arr.shape != ref_shape:
        from skimage.transform import resize
        arr = resize(arr, ref_shape, anti_aliasing=True, preserve_range=True)
    normed = pct_normalize(arr)
    cci_array += (normed * weight).astype('float32')
    print(f'  {name}: weight={weight:.2f}  mean={np.nanmean(normed):.3f}')

# Wrap in DataArray (inherit spatial coords from NDVI change)
cci_da = ndvi_change.copy(data=cci_array).rename('CCI')
print(f'\nCCI range: {float(cci_da.min()):.3f} → {float(cci_da.max()):.3f}')
print(f'CCI mean : {float(cci_da.mean()):.3f}')

"""## 8. Write Rasters to GeoTIFF for Zonal Stats"""

import os
import tempfile

RASTER_DIR = '/content/rasters'
os.makedirs(RASTER_DIR, exist_ok=True)


def write_geotiff(da, path, crs='EPSG:32618'):
    """Write a 2-D xarray DataArray to a single-band GeoTIFF."""
    da_rio = da.copy()
    da_rio = da_rio.rio.write_crs(crs)
    da_rio.rio.to_raster(path, driver='GTiff', compress='lzw')
    return path


# Write all layers we'll use for zonal stats
layers_to_write = {
    'CCI':          cci_da,
    'NDVI_change':  ndvi_change,
    'NDBI_change':  ndbi_change,
    'BSI_change':   bsi_change,
    'VV_logratio':  vv_logratio,
    'NDVI_pre':     idx_pre['NDVI'],
    'NDVI_post':    idx_post['NDVI'],
    'NDBI_pre':     idx_pre['NDBI'],
    'NDBI_post':    idx_post['NDBI'],
    'BSI_pre':      idx_pre['BSI'],
    'BSI_post':     idx_post['BSI'],
    'VV_pre':       s1_pre.sel(band='vv'),
    'VV_post':      s1_post.sel(band='vv'),
}

raster_paths = {}
for name, da in layers_to_write.items():
    path = f'{RASTER_DIR}/{name}.tif'
    write_geotiff(da, path)
    raster_paths[name] = path
    print(f'  Written: {path}')

print('\nAll rasters written.')

"""## 9. Zonal Statistics — Score Each Parcel"""

# Reproject parcels to match raster CRS (UTM 18N)
parcels_utm = parcels_gdf.to_crs('EPSG:32618')

stats_records = []

for name, tif_path in raster_paths.items():
    result = zonal_stats(
        parcels_utm,
        tif_path,
        stats=['mean', 'std', 'min', 'max'],
        nodata=np.nan,
        all_touched=True
    )
    for i, r in enumerate(result):
        if i >= len(stats_records):
            stats_records.append({'parcel_id': parcels_gdf.iloc[i]['parcel_id']})
        stats_records[i][name] = r.get('mean', np.nan)
        stats_records[i][f'{name}_std'] = r.get('std', np.nan)

stats_df = pd.DataFrame(stats_records)

# Flag parcels
stats_df['construction_flag'] = stats_df['CCI'] >= CCI_THRESHOLD
stats_df['CCI_class'] = pd.cut(
    stats_df['CCI'],
    bins=[0, 0.25, 0.40, 0.55, 0.70, 1.0],
    labels=['None', 'Low', 'Moderate', 'High', 'Very High']
)

# Merge back to spatial GDF
results_gdf = parcels_gdf.merge(stats_df, on='parcel_id', how='left')
results_gdf.to_file(OUTPUT_GEOJSON, driver='GeoJSON')

print(f'Parcel stats computed for {len(stats_df)} parcels.')
print(f'Construction flagged (CCI ≥ {CCI_THRESHOLD}): {stats_df["construction_flag"].sum()}')
display(stats_df[['parcel_id','CCI','NDVI_change','NDBI_change',
                   'BSI_change','VV_logratio','CCI_class','construction_flag']]
        .sort_values('CCI', ascending=False))

"""## 10. Diagnostic Charts"""

fig, axes = plt.subplots(2, 3, figsize=(16, 9))
fig.patch.set_facecolor('#0f1117')
fig.suptitle('Construction Change Detection — Parcel Diagnostics',
             fontsize=14, fontweight='bold', color='white', y=1.01)

for ax in axes.flat:
    ax.set_facecolor('#1a1d27')
    for spine in ax.spines.values():
        spine.set_edgecolor('#333')
    ax.tick_params(colors='#aaa')
    ax.xaxis.label.set_color('#ccc')
    ax.yaxis.label.set_color('#ccc')
    ax.title.set_color('white')

ACCENT = '#e07b39'
FLAG_COLOR = '#c0392b'
OK_COLOR = '#2980b9'

# 1. CCI distribution
ax = axes[0, 0]
vals = stats_df['CCI'].dropna()
ax.hist(vals, bins=15, color=ACCENT, alpha=0.8, edgecolor='#222')
ax.axvline(CCI_THRESHOLD, color='red', linestyle='--', linewidth=1.5,
           label=f'Threshold ({CCI_THRESHOLD})')
ax.set_title('CCI Distribution')
ax.set_xlabel('CCI Score')
ax.legend(labelcolor='white', facecolor='#1a1d27', edgecolor='#444')

# 2. CCI bar by parcel
ax = axes[0, 1]
sdf = stats_df.sort_values('CCI', ascending=False).dropna(subset=['CCI'])
colors = [FLAG_COLOR if f else OK_COLOR for f in sdf['construction_flag']]
ax.bar(sdf['parcel_id'], sdf['CCI'], color=colors, edgecolor='#222')
ax.axhline(CCI_THRESHOLD, color='white', linestyle='--', linewidth=1)
ax.set_title('CCI by Parcel  (red = flagged)')
ax.set_xlabel('Parcel ID')
ax.tick_params(axis='x', rotation=45)

# 3. NDVI pre vs post
ax = axes[0, 2]
sc = ax.scatter(stats_df['NDVI_pre'], stats_df['NDVI_post'],
                c=stats_df['CCI'], cmap='RdYlGn_r', s=90, edgecolors='white',
                linewidths=0.5, vmin=0, vmax=1)
lims = [min(stats_df[['NDVI_pre','NDVI_post']].min()),
        max(stats_df[['NDVI_pre','NDVI_post']].max())]
ax.plot(lims, lims, '--', color='#555', linewidth=1)
ax.set_xlabel('NDVI Pre')
ax.set_ylabel('NDVI Post')
ax.set_title('NDVI Pre vs Post')
plt.colorbar(sc, ax=ax, label='CCI')

# 4. BSI pre vs post
ax = axes[1, 0]
sc2 = ax.scatter(stats_df['BSI_pre'], stats_df['BSI_post'],
                 c=stats_df['CCI'], cmap='RdYlGn_r', s=90, edgecolors='white',
                 linewidths=0.5, vmin=0, vmax=1)
ax.plot([-1,1],[-1,1],'--',color='#555',linewidth=1)
ax.set_xlabel('BSI Pre')
ax.set_ylabel('BSI Post')
ax.set_title('Bare Soil Index Pre vs Post')
plt.colorbar(sc2, ax=ax, label='CCI')

# 5. SAR VV pre vs post
ax = axes[1, 1]
sc3 = ax.scatter(stats_df['VV_pre'], stats_df['VV_post'],
                 c=stats_df['CCI'], cmap='RdYlGn_r', s=90, edgecolors='white',
                 linewidths=0.5, vmin=0, vmax=1)
ax.plot([-30,5],[-30,5],'--',color='#555',linewidth=1)
ax.set_xlabel('VV Pre (dB)')
ax.set_ylabel('VV Post (dB)')
ax.set_title('SAR VV Backscatter Pre vs Post')
plt.colorbar(sc3, ax=ax, label='CCI')

# 6. Feature correlations
ax = axes[1, 2]
corr_cols = ['CCI','NDVI_change','NDBI_change','BSI_change','VV_logratio']
corr = stats_df[corr_cols].dropna().corr()
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', center=0,
            ax=ax, square=True, linewidths=0.5,
            annot_kws={'color':'white','size':8})
ax.set_title('Feature Correlations')

plt.tight_layout()
plt.savefig('/content/cci_diagnostics.png', dpi=150, bbox_inches='tight',
            facecolor='#0f1117')
plt.show()
print('Chart saved → /content/cci_diagnostics.png')

"""## 11. Raster Preview — CCI + Key Change Layers"""

fig, axes = plt.subplots(2, 3, figsize=(16, 9))
fig.patch.set_facecolor('#0f1117')
fig.suptitle('Raster Change Layers', fontsize=13, color='white', fontweight='bold')

plots = [
    (cci_da,       'Construction Change Index (CCI)',    'hot',         0,    1),
    (ndvi_change,  'NDVI Change (post − pre)',           'RdYlGn',     -0.4,  0.4),
    (ndbi_change,  'NDBI Change (post − pre)',           'RdBu_r',     -0.3,  0.3),
    (bsi_change,   'BSI Change (post − pre)',            'RdBu_r',     -0.3,  0.3),
    (vv_logratio,  'SAR VV Log-Ratio (dB)',              'PuOr_r',     -6,    6),
    (vv_variance,  'SAR VV Temporal Variance',           'YlOrRd',      0,   None),
]

for ax, (da, title, cmap, vmin, vmax) in zip(axes.flat, plots):
    arr = da.values if hasattr(da, 'values') else da
    kw = dict(cmap=cmap, vmin=vmin)
    if vmax is not None:
        kw['vmax'] = vmax
    im = ax.imshow(arr, **kw)
    ax.set_title(title, color='white', fontsize=9)
    ax.axis('off')
    plt.colorbar(im, ax=ax, shrink=0.7)

plt.tight_layout()
plt.savefig('/content/raster_panels.png', dpi=150, bbox_inches='tight',
            facecolor='#0f1117')
plt.show()
print('Raster panel saved → /content/raster_panels.png')

"""## 12. Interactive Folium Map"""

import rasterio
from rasterio.warp import calculate_default_transform, reproject as rio_reproject, Resampling as RioResampling
import folium
from folium import raster_layers

# Reproject CCI to EPSG:4326 for Folium overlay
cci_tif_utm = raster_paths['CCI']
cci_tif_wgs = '/content/rasters/CCI_wgs84.tif'

with rasterio.open(cci_tif_utm) as src:
    transform, width, height = calculate_default_transform(
        src.crs, 'EPSG:4326', src.width, src.height, *src.bounds
    )
    meta = src.meta.copy()
    meta.update({'crs': 'EPSG:4326', 'transform': transform,
                 'width': width, 'height': height, 'nodata': np.nan})
    with rasterio.open(cci_tif_wgs, 'w', **meta) as dst:
        rio_reproject(
            source=rasterio.band(src, 1),
            destination=rasterio.band(dst, 1),
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=transform,
            dst_crs='EPSG:4326',
            resampling=RioResampling.bilinear
        )

# Build folium map
lon_min, lat_min, lon_max, lat_max = BBOX
center = [(lat_min + lat_max) / 2, (lon_min + lon_max) / 2]

m = folium.Map(location=center, zoom_start=14, tiles='CartoDB dark_matter')

# CCI raster overlay
with rasterio.open(cci_tif_wgs) as src:
    bounds_wgs = src.bounds
    cci_arr = src.read(1)

# Colorize CCI to RGBA PNG for overlay
import io, base64
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from PIL import Image as PILImage

norm = Normalize(vmin=0, vmax=1)
cmap_fn = plt.cm.get_cmap('hot')
rgba = cmap_fn(norm(np.where(np.isnan(cci_arr), -1, cci_arr)))
rgba[np.isnan(cci_arr)] = [0, 0, 0, 0]  # transparent NaN
img_pil = PILImage.fromarray((rgba * 255).astype(np.uint8), mode='RGBA')
buf = io.BytesIO()
img_pil.save(buf, format='PNG')
img_b64 = base64.b64encode(buf.getvalue()).decode('utf-8')

folium.raster_layers.ImageOverlay(
    image=f'data:image/png;base64,{img_b64}',
    bounds=[[bounds_wgs.bottom, bounds_wgs.left], [bounds_wgs.top, bounds_wgs.right]],
    opacity=0.65,
    name='CCI Raster'
).add_to(m)

# Parcel choropleth + tooltips
def cci_color(v):
    if pd.isna(v): return '#555555'
    if v >= 0.70:  return '#c0392b'
    if v >= 0.50:  return '#e67e22'
    if v >= 0.35:  return '#f1c40f'
    return '#27ae60'

for _, row in results_gdf.iterrows():
    color = cci_color(row.get('CCI', np.nan))
    cci_val = row.get('CCI', np.nan)
    folium.GeoJson(
        row.geometry.__geo_interface__,
        style_function=lambda x, c=color: {
            'fillColor': c, 'color': 'white',
            'weight': 1.5, 'fillOpacity': 0.4
        },
        tooltip=folium.Tooltip(
            f"<b>Parcel:</b> {row.get('parcel_id', '?')}<br>"
            f"<b>CCI:</b> {cci_val:.3f}<br>"
            f"<b>Class:</b> {row.get('CCI_class', '?')}<br>"
            f"<b>NDVI Δ:</b> {row.get('NDVI_change', np.nan):.3f}<br>"
            f"<b>BSI Δ:</b> {row.get('BSI_change', np.nan):.3f}<br>"
            f"<b>VV Δ dB:</b> {row.get('VV_logratio', np.nan):.2f}<br>"
            f"<b>🏗 Flagged:</b> {'Yes' if row.get('construction_flag') else 'No'}"
        )
    ).add_to(m)

# Legend
legend_html = '''
<div style="position:fixed;bottom:30px;left:30px;z-index:9999;
            background:#1a1a2e;padding:12px 16px;border-radius:8px;
            border:1px solid #444;font-family:monospace;color:white;font-size:12px;">
<b>CCI Score</b><br>
<span style="color:#c0392b">■</span> Very High ≥ 0.70<br>
<span style="color:#e67e22">■</span> High &nbsp;&nbsp;&nbsp;0.50–0.70<br>
<span style="color:#f1c40f">■</span> Moderate 0.35–0.50<br>
<span style="color:#27ae60">■</span> Low/None &lt; 0.35
</div>'''
m.get_root().html.add_child(folium.Element(legend_html))
folium.LayerControl().add_to(m)

m.save(OUTPUT_MAP)
print(f'Interactive map saved → {OUTPUT_MAP}')
m

import rasterio
from rasterio.warp import calculate_default_transform, reproject as rio_reproject, Resampling as RioResampling
import folium
from folium import raster_layers

# --- BBOX / CENTER ---
lon_min, lat_min, lon_max, lat_max = BBOX
center = [(lat_min + lat_max) / 2, (lon_min + lon_max) / 2]

# --- SATELLITE / NAIP-LIKE BASEMAP ---
# Replace with your preferred/authorized satellite or NAIP tiles.
SAT_TILES = "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"

m = folium.Map(location=center, zoom_start=16, tiles=None)

folium.TileLayer(
    tiles=SAT_TILES,
    attr="Esri World Imagery",
    name="Satellite",
    overlay=False,
    control=True,
).add_to(m)

# Optional alternate base
folium.TileLayer(
    tiles="CartoDB dark_matter",
    name="Dark",
    overlay=False,
    control=True,
).add_to(m)

# --- REPROJECT CCI TO EPSG:4326 ---
cci_tif_utm = raster_paths['CCI']
cci_tif_wgs = '/content/rasters/CCI_wgs84.tif'

with rasterio.open(cci_tif_utm) as src:
    transform, width, height = calculate_default_transform(
        src.crs, 'EPSG:4326', src.width, src.height, *src.bounds
    )
    meta = src.meta.copy()
    meta.update({
        'crs': 'EPSG:4326',
        'transform': transform,
        'width': width,
        'height': height,
        'nodata': np.nan,
    })
    with rasterio.open(cci_tif_wgs, 'w', **meta) as dst:
        rio_reproject(
            source=rasterio.band(src, 1),
            destination=rasterio.band(dst, 1),
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=transform,
            dst_crs='EPSG:4326',
            resampling=RioResampling.bilinear,
        )

# --- CCI RASTER OVERLAY ---
with rasterio.open(cci_tif_wgs) as src:
    bounds_wgs = src.bounds
    cci_arr = src.read(1)

import io, base64
from matplotlib.colors import Normalize
import matplotlib.pyplot as plt
from PIL import Image as PILImage

norm = Normalize(vmin=0, vmax=1)
cmap_fn = plt.cm.get_cmap('hot')
rgba = cmap_fn(norm(np.where(np.isnan(cci_arr), -1, cci_arr)))
rgba[np.isnan(cci_arr)] = [0, 0, 0, 0]

img_pil = PILImage.fromarray((rgba * 255).astype(np.uint8), mode='RGBA')
buf = io.BytesIO()
img_pil.save(buf, format='PNG')
img_b64 = base64.b64encode(buf.getvalue()).decode('utf-8')

folium.raster_layers.ImageOverlay(
    image=f"data:image/png;base64,{img_b64}",
    bounds=[[bounds_wgs.bottom, bounds_wgs.left],
            [bounds_wgs.top,    bounds_wgs.right]],
    opacity=0.65,
    name="CCI Raster",
).add_to(m)

# --- PARCELS LAYER ---
def cci_color(v):
    if pd.isna(v): return '#555555'
    if v >= 0.70:  return '#c0392b'
    if v >= 0.50:  return '#e67e22'
    if v >= 0.35:  return '#f1c40f'
    return '#27ae60'

for _, row in results_gdf.iterrows():
    color = cci_color(row.get('CCI', np.nan))
    cci_val = row.get('CCI', np.nan)
    folium.GeoJson(
        row.geometry.__geo_interface__,
        style_function=lambda x, c=color: {
            'fillColor': c,
            'color': 'white',
            'weight': 1.5,
            'fillOpacity': 0.4,
        },
        tooltip=folium.Tooltip(
            f"<b>Parcel:</b> {row.get('parcel_id', '?')}<br>"
            f"<b>CCI:</b> {cci_val:.3f}<br>"
            f"<b>Class:</b> {row.get('CCI_class', '?')}<br>"
            f"<b>NDVI Δ:</b> {row.get('NDVI_change', np.nan):.3f}<br>"
            f"<b>BSI Δ:</b> {row.get('BSI_change', np.nan):.3f}<br>"
            f"<b>VV Δ dB:</b> {row.get('VV_logratio', np.nan):.2f}<br>"
            f"<b>🏗 Flagged:</b> {'Yes' if row.get('construction_flag') else 'No'}"
        ),
        name="Parcels",
    ).add_to(m)

# --- LEGEND + CONTROLS ---
legend_html = '''
<div style="position:fixed;bottom:30px;left:30px;z-index:9999;
            background:#1a1a2e;padding:12px 16px;border-radius:8px;
            border:1px solid #444;font-family:monospace;color:white;font-size:12px;">
<b>CCI Score</b><br>
<span style="color:#c0392b">■</span> Very High ≥ 0.70<br>
<span style="color:#e67e22">■</span> High &nbsp;&nbsp;&nbsp;0.50–0.70<br>
<span style="color:#f1c40f">■</span> Moderate 0.35–0.50<br>
<span style="color:#27ae60">■</span> Low/None &lt; 0.35
</div>'''
m.get_root().html.add_child(folium.Element(legend_html))

folium.LayerControl(collapsed=False).add_to(m)

m.save(OUTPUT_MAP)
print(f"Interactive map saved → {OUTPUT_MAP}")
m

"""## 13. Optional — Export All Outputs"""

# Download everything from Colab to local machine
from google.colab import files

files.download(OUTPUT_GEOJSON)          # parcel scores GeoJSON
files.download(OUTPUT_MAP)              # interactive HTML map
files.download('/content/cci_diagnostics.png')
files.download('/content/raster_panels.png')

# Optional: zip and download all GeoTIFFs
import shutil
shutil.make_archive('/content/rasters_export', 'zip', RASTER_DIR)
files.download('/content/rasters_export.zip')

"""## 14. Notes & Next Steps

**UTM zone:** Default is 32618 (UTM 18N, US Mid-Atlantic). Change `epsg=` in both `load_s2_stack` and `load_s1_stack` for other regions:  
- West Coast US → 32610  
- UK/Europe → 32630  
- Use https://epsg.io to look up your zone

**Scaling up:** For large AOIs, increase Dask chunk size in stackstac and avoid `.compute()` until final zonal stats step. Consider Google Colab Pro for RAM.

**Real parcel data sources:**
- Maryland: https://geodata.md.gov
- National (OpenStreetMap buildings): `osmnx.geometries_from_bbox()`
- Any county GIS portal → download parcel shapefile → set `PARCEL_INPUT = 'shapefile'`

**Add a classifier:** Once you have labeled parcels (permitted vs not), swap the weighted CCI for a scikit-learn Random Forest trained on the extracted feature columns.

**Time series:** Loop over monthly date windows to produce a CCI time series per parcel — useful for detecting construction *start* and *completion* dates.
"""
