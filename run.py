# %%
from deepspt_src import *  
import torch  
  
# 1. Load your CSV data  
datapath = '/home/S-BJ/ARTEMIS2D/trackingresults/MAX_20260430_median1_meta_normtimeaug5_tracks.geff/'  
filename_X = 'output.csv'  
X_to_eval = load_X(datapath, filename_X, features=['XYZ', 'SL', 'DP'])  

# %%
# 2. Create dummy labels (required for prediction)  
y_to_eval = [np.ones(len(x))*0.5 for x in X_to_eval]  

# %%
# 3. Find and load the model from mlruns/3  
path = 'mlruns/3'  
best_models_sorted = find_models_for_from_path(path)  
model = load_UnetModels_directly(best_models_sorted[0], device='cpu', dim=2)  

# %%
# 4. Run predictions  
results = make_preds(model, X_to_eval, y_to_eval, device='cpu')  
predictions = results['masked_pred']
# %%

data = []

for i in range(len(predictions)):
    x = X_to_eval[i]
    pred = predictions[i]
    data.append(pd.DataFrame({
        "track_id": [i + 1] * len(x),
        "y": x[:, 1],
        "x": x[:, 0],
        "t": list(range(len(x))),
        "pred": pred,
    }))

df = pd.concat(data)
# %%
df.to_csv("predictions.csv", index=False)
# %%
