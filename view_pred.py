# %%
import pandas as pd
import napari
from tifffile import imread

path = "/home/S-BJ/DeepSPT/predictions.csv"
image_path = "/home/S-BJ/ARTEMIS2D/trackingresults/MAX_20260430_median1_meta_norm.ome.tif"
image = imread(image_path)

df = pd.read_csv(path)

# viewer = napari.Viewer()
# layer = viewer.add_tracks(df[["track_id", "t", "y", "x"]], properties={"pred": df["pred"]})
# layer.color_by = "pred"
# colormap="hsv"
# viewer.add_image(image)

viewer = napari.Viewer()

layer = viewer.add_tracks(
    df[["track_id", "t", "y", "x"]].to_numpy(),
    properties={"pred": df["pred"].to_numpy()},
    color_by="pred",
    colormap="viridis",
    tail_width=2,
)

viewer.add_image(image)

# %%
cls_map = {0: "normal", 1: "directed", 2: "confined", 3: "subdiffusive"}

# %%
for p, grp in df.groupby("pred"):
    print(f"Pred ({p})", cls_map[p])
    print(grp["track_id"].unique())


# %%
