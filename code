"""
Deforestation_Monitoring_System_GLOBAL.py

Upgraded from a regional demo to a world-scale simulator.
- Single-region monitoring only (no hotspots)
- Heatmap + tile overlays in Folium
- Batch stats + CSV export removed (since single region)
- Pluggable data layer (simulated NDVI; placeholders for real APIs)
- Lightweight controls for Colab/Jupyter

Note: Still uses simulated imagery/NDVI. Replace `DATA SOURCE HOOKS` with real satellite API calls (Sentinel Hub, Google Earth Engine, NASA, Planet).
"""
# ================================
# Imports
# ================================

import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image, ImageDraw
from datetime import datetime, timedelta
import folium
from folium.plugins import MiniMap, MousePosition
from IPython.display import display
import pandas as pd
from skimage import measure, morphology
import plotly.graph_objects as go
import ipywidgets as widgets
from sklearn.metrics import precision_score, recall_score, f1_score, jaccard_score

plt.rcParams.update({"figure.facecolor": "white", "axes.facecolor": "white"})

print("🌍 Single-Region Deforestation Monitoring — SIMULATED")
print("This notebook monitors a selected region using simulated NDVI.")

# ================================
# Generate sample satellite-like images
# ================================

def generate_sample_image(bounds, vegetation_factor, date):
    width, height = 256, 256
    img = Image.new('RGB', (width, height), (100, 100, 100))
    draw = ImageDraw.Draw(img)

    # vegetation patches
    num_patches = int(max(1, vegetation_factor * 25))
    for _ in range(num_patches):
        x, y = np.random.randint(0, width), np.random.randint(0, height)
        size = np.random.randint(10, 80)
        green_intensity = int(np.clip(vegetation_factor * 170 + 40, 0, 255))
        draw.ellipse([x, y, x+size, y+size], fill=(0, green_intensity, 0))

    # brown lines mimic roads/clearings when vegetation lower
    if vegetation_factor < 0.7:
        for _ in range(int((1-vegetation_factor) * 6)):
            x1, y1 = np.random.randint(0, width), np.random.randint(0, height)
            x2, y2 = np.random.randint(0, width), np.random.randint(0, height)
            draw.line([x1, y1, x2, y2], fill=(139, 69, 19), width=3)

    draw.text((6, height-16), f"Sim: {date:%Y-%m-%d}", fill=(255, 255, 255))
    return img

# ================================
# Simulated NDVI extraction
# ================================

def calculate_ndvi(image):
    arr = np.array(image).astype(float)
    greenness = arr[:, :, 1] / 255.0
    noise = np.random.normal(0, 0.08, greenness.shape)
    ndvi = np.clip(greenness + noise, -1, 1)
    return ndvi

# ================================
# Change detection & mask creation
# ================================

def detect_changes(ndvi1, ndvi2, threshold=0.2):
    change = ndvi2 - ndvi1
    mask = change < -threshold
    return mask, change

# ================================
# Analyze deforestation patterns
# ================================

def analyze_deforestation_patterns(mask, min_size=10):
    cleaned = morphology.remove_small_objects(mask, min_size=min_size)
    cleaned = morphology.remove_small_holes(cleaned, area_threshold=min_size)
    labeled = measure.label(cleaned)
    regions = measure.regionprops(labeled)
    num_patches = len(regions)
    patch_sizes = [r.area for r in regions]
    total_area = int(np.sum(cleaned))
    return cleaned, labeled, regions, num_patches, patch_sizes, total_area

# ================================
# Accuracy evaluation (simulated)
# ================================

def evaluate_accuracy(results):
    y_true_all, y_pred_all = [], []

    for res in results:
        if "gt_mask" not in res:
            continue
        gt = res["gt_mask"].flatten().astype(int)
        pred = res["mask"].flatten().astype(int)
        y_true_all.extend(gt)
        y_pred_all.extend(pred)

    if not y_true_all:
        print("⚠️ No ground-truth available (real-data mode).")
        return None

    precision = precision_score(y_true_all, y_pred_all, zero_division=0)
    recall = recall_score(y_true_all, y_pred_all, zero_division=0)
    f1 = f1_score(y_true_all, y_pred_all, zero_division=0)
    iou = jaccard_score(y_true_all, y_pred_all, zero_division=0)

    print("\n📊 Accuracy Evaluation (Simulation):")
    print(f"Precision: {precision:.3f}")
    print(f"Recall:    {recall:.3f}")
    print(f"F1 Score:  {f1:.3f}")
    print(f"IoU:       {iou:.3f}")

    return {"precision": precision, "recall": recall, "f1": f1, "iou": iou}

# ================================
# NDVI time series simulation
# ================================

def generate_time_series_data(start_date, end_date):

    if isinstance(start_date, datetime):
        start_date = start_date.date()
    if isinstance(end_date, datetime):
        end_date = end_date.date()

    dates, values = [], []
    cur = start_date
    while cur <= end_date:
        days = (cur - start_date).days
        seasonal = 0.28 * np.sin(2 * np.pi * days / 365)
        trend = -0.0012 * days
        noise = np.random.normal(0, 0.04)
        v = max(0.25, 0.72 + seasonal + trend + noise)
        dates.append(cur)
        values.append(v)
        cur += timedelta(days=30)
    return dates, values


# ================================
# Simulate deforestation for a region
# ================================

def simulate_deforestation_for_tile(bounds, start_date, end_date, ndvi_threshold=0.2, min_patch=10):
    lat_min, lon_min, lat_max, lon_max = bounds
    band_mid_lat = (lat_min + lat_max) / 2
    tropical_weight = max(0, 1 - abs(band_mid_lat) / 40)

    veg_start = 0.75 * (0.6 + 0.4 * tropical_weight)
    veg_end = veg_start - (0.15 + 0.25 * tropical_weight) * np.random.uniform(0.6, 1.0)

    img1 = generate_sample_image(bounds, veg_start, start_date)
    img2 = generate_sample_image(bounds, veg_end, end_date)
    ndvi1 = calculate_ndvi(img1)
    ndvi2 = calculate_ndvi(img2)
    mask, change = detect_changes(ndvi1, ndvi2, ndvi_threshold)
    cleaned, labeled, regions, n_patches, patch_sizes, total_area = analyze_deforestation_patterns(mask, min_patch)

    total_pixels = mask.size
    pct = (np.sum(cleaned) / total_pixels) * 100
    gt_mask = mask.copy()
    return {
        "bounds": bounds,
        "center": [(lat_min+lat_max)/2, (lon_min+lon_max)/2],
        "total_pixels": int(total_pixels),
        "deforested_pixels": int(np.sum(cleaned)),
        "deforestation_pct": float(pct),
        "num_patches": int(n_patches),
        "avg_patch": float(np.mean(patch_sizes)) if n_patches else 0.0,
        "max_patch": int(max(patch_sizes)) if n_patches else 0,
        "img_start": img1,
        "img_end": img2,
        "change": change,
        "mask": cleaned,
        "gt_mask": gt_mask,
    }
# ================================
# Save results to CSV
# ================================
def save_results_to_csv(res, filename="deforestation_results.csv"):
    df = pd.DataFrame([{
        "region": region_dropdown.value,
        "start_date": start_date_picker.value.strftime("%Y-%m-%d"),
        "end_date": end_date_picker.value.strftime("%Y-%m-%d"),
        "deforested_pixels": res["deforested_pixels"],
        "total_pixels": res["total_pixels"],
        "deforestation_pct": res["deforestation_pct"],
        "num_patches": res["num_patches"],
        "avg_patch": res["avg_patch"],
        "max_patch": res["max_patch"],
    }])

    # Append if file exists, otherwise create new
    if os.path.exists(filename):
        df.to_csv(filename, mode="a", header=False, index=False)
    else:
        df.to_csv(filename, index=False)

# ================================
# Region selection
# ================================

region_options = {
    "Amazon Rainforest (Sample)": {"center": [-3.4653, -62.2159], "bounds": [-5, -64, -2, -60]},
    "Congo Basin (Sample)": {"center": [0.8327, 21.5565], "bounds": [-1, 19, 2, 24]},
    "Borneo (Sample)": {"center": [0.9619, 114.5548], "bounds": [-2, 112, 3, 117]},
    "Madagascar (Sample)": {"center": [-18.7669, 46.8691], "bounds": [-21, 44, -17, 48]},
    "New Guinea (Sample)": {"center": [-5.0, 141.0], "bounds": [-7, 139, -3, 143]},
    "Southeast Asia (Mekong)": {"center": [13.0, 105.0], "bounds": [11, 103, 15, 107]},
    "Eastern Himalayas": {"center": [27.5, 91.0], "bounds": [26, 90, 29, 92]},
    "West Africa (Ivory Coast)": {"center": [7.5, -5.5], "bounds": [6, -7, 9, -4]},
    "Central America (Guatemala)": {"center": [16.0, -90.0], "bounds": [15, -91, 17, -89]},
    "Brazil – Mato Grosso": {"center": [-13.0, -56.0], "bounds": [-14, -58, -12, -54]},
    "Peru – Madre de Dios": {"center": [-12.5, -70.0], "bounds": [-13, -71, -12, -69]},
    "India – Northeast": {"center": [25.5, 93.0], "bounds": [24, 92, 27, 94]},
}

region_dropdown = widgets.Dropdown(
    options=list(region_options.keys()),
    value="Amazon Rainforest (Sample)",
    description='Region:',
)

# ================================
# Run button and parameters
# ================================

start_date_picker = widgets.DatePicker(
    description='Start Date',
    value=datetime.now() - timedelta(days=365),
)
end_date_picker = widgets.DatePicker(
    description='End Date',
    value=datetime.now(),
)
ndvi_threshold_slider = widgets.FloatSlider(
    value=0.1, min=0.05, max=0.5, step=0.05, description='NDVI Δ thr'
)
min_patch_slider = widgets.IntSlider(
    value=20, min=1, max=200, step=5, description='Min patch px'
)
run_button = widgets.Button(
    description='Run Monitoring',
    button_style='success',
    icon='play'
)

print("Select parameters:")
display(region_dropdown)
display(start_date_picker)
display(end_date_picker)
display(widgets.HBox([ndvi_threshold_slider, min_patch_slider]))
display(run_button)

# ================================
# Folium helpers
# ================================

def add_tile_rectangles(m, results, pct_threshold=5.0):
    for r in results:
        lat_min, lon_min, lat_max, lon_max = r["bounds"]
        color = '#ff0000' if r["deforestation_pct"] >= pct_threshold else '#ffa500'
        folium.Rectangle(
            bounds=[(lat_min, lon_min), (lat_max, lon_max)],
            color=color,
            weight=1,
            fill=True,
            fill_color=color,
            fill_opacity=min(0.6, r["deforestation_pct"]/25 + 0.1),
            tooltip=(f"Δ Forest: {r['deforestation_pct']:.2f}%\n"
                     f"Patches: {r['num_patches']}\n"
                     f"Bounds: {lat_min:.2f},{lon_min:.2f} → {lat_max:.2f},{lon_max:.2f}")
        ).add_to(m)

# ================================
# Run monitoring
# ================================

def on_run(_):
    print("\n🌿 Running monitoring…")
    sel = region_options[region_dropdown.value]
    start_date = start_date_picker.value
    end_date = end_date_picker.value
    ndvi_threshold = ndvi_threshold_slider.value
    min_patch = min_patch_slider.value

    res = simulate_deforestation_for_tile(sel["bounds"], start_date, end_date, ndvi_threshold, min_patch)

    save_results_to_csv(res)
    print("✅ Results saved to deforestation_results.csv")

    # Map
    m = folium.Map(location=sel["center"], zoom_start=6, tiles="CartoDB positron")
    MiniMap().add_to(m)
    MousePosition().add_to(m)
    add_tile_rectangles(m, [res], pct_threshold=5.0)

    # Legend
    legend_html = '''
     <div style="position: fixed;
                 bottom: 30px; left: 30px; width: 220px; height: 130px;
                 background-color: white; border:2px solid grey; z-index:9999; font-size:14px;">
     &nbsp;<b>Deforestation Risk Levels</b><br>
     &nbsp;<i style="background:#00f; color:white">&nbsp;&nbsp;&nbsp;</i>&nbsp; Low (0.0 - 0.3)<br>
     &nbsp;<i style="background:#0f0;">&nbsp;&nbsp;&nbsp;</i>&nbsp; Moderate (0.3 - 0.6)<br>
     &nbsp;<i style="background:#ff0;">&nbsp;&nbsp;&nbsp;</i>&nbsp; High (0.6 - 0.8)<br>
     &nbsp;<i style="background:#f00; color:white">&nbsp;&nbsp;&nbsp;</i>&nbsp; Severe (0.8 - 1.0)<br>
     </div>
     '''
    m.get_root().html.add_child(folium.Element(legend_html))
    display(m)

    # Visuals
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes[0,0].imshow(res["img_start"]); axes[0,0].set_title(f"Start: {start_date:%Y-%m-%d}"); axes[0,0].axis('off')
    axes[0,1].imshow(res["img_end"]); axes[0,1].set_title(f"End: {end_date:%Y-%m-%d}"); axes[0,1].axis('off')
    ch = axes[1,0].imshow(res["change"], vmin=-1, vmax=1, cmap='RdYlGn'); axes[1,0].set_title("NDVI Change"); axes[1,0].axis('off'); plt.colorbar(ch, ax=axes[1,0])
    mask_rgb = np.zeros((*res["img_end"].size[::-1], 3), dtype=np.uint8)
    mask_rgb[res["mask"]] = [255,0,0]
    axes[1,1].imshow(res["img_end"]); axes[1,1].imshow(mask_rgb, alpha=0.45); axes[1,1].set_title("Deforestation (red)"); axes[1,1].axis('off')
    plt.tight_layout(); plt.show()

    evaluate_accuracy([res])

    # Time series
    dates, vals = generate_time_series_data(start_date, end_date)
    fig_ts = go.Figure(); fig_ts.add_trace(go.Scatter(x=dates, y=vals, mode='lines+markers', name='Vegetation Index'))
    fig_ts.add_hline(y=0.5, line_dash="dash", annotation_text="Danger Threshold")
    fig_ts.update_layout(title="Vegetation Index Time Series", xaxis_title="Date", yaxis_title="Index", hovermode="x unified")
    fig_ts.show()

    pct = res['deforestation_pct']
    print("\n📊 Summary — Single Region")
    print("===========================")
    print(f"Deforested pixels: {res['deforested_pixels']} / {res['total_pixels']} ({pct:.2f}%)")
    print(f"Patches: {res['num_patches']} | Avg patch: {res['avg_patch']:.2f} | Max patch: {res['max_patch']}")
    if pct > 5:
        print("\n⚠️  Significant deforestation detected in this region.")

run_button.on_click(on_run)

print("\n" + "="*64)
print("HOW THIS SINGLE-REGION MONITOR WORKS (Simulated)")
print("="*64)
print(
    """
1) The selected region is analyzed for two dates (start/end).
2) Simulated NDVI is computed, negative changes detected.
3) Deforestation patches are identified and visualized via Folium + matplotlib.
4) Time series plots vegetation trends.
NEXT: Swap simulated imagery with real data from Sentinel, GEE, NASA LP DAAC, Planet NICFI, etc.
"""
)
