# %%
from deepspt_src import *  
import torch
import pandas as pd
from dataclasses import dataclass
from pathlib import Path
import numpy as np

# %%
@dataclass
class Track:
    start_x: float
    start_y: float
    timepoints: np.ndarray
    track_id: int
    features: np.ndarray

# %%
# 1. Load your CSV data  
datapath = '/home/S-BJ/ARTEMIS2D/trackingresults/MAX_20260430_median1_meta_normtimeaug5_tracks.geff/'  
filename_X = 'output.csv'

file = Path(datapath) / filename_X
dffile = pd.read_csv(file)

print(dffile.head())
print(f"unproceesed file preview :{dffile.columns}")

tracks = [] # initialize a tracks list that will contain the tracks of each cell

def load_X_modified(datapath, filename_X, features_list=['XYZ','SL', 'DP']):
        # load in the csv
        file = Path(datapath) / filename_X
        df = pd.read_csv(file) # read the file and make it pd df
        

        for track_id, track_df in df.groupby("particle"): # keeps trakcs and cells paired # one little table per track
            if len(track_df) == 1:
                 continue
            
            track_df = track_df.sort_values("frame") #  sorts by time
            # after this we have a df where each cell and track ares seperate

            print("track id:", track_id)
            print(track_df.head())

            # take the current df and extract the time for each cell track
            x = track_df["x"].to_numpy() # extract track x values
            y = track_df["y"].to_numpy() # extract track y values
            timepoints = track_df["frame"].to_numpy() # minitable is now arrays

            features = track_df[["x", "y"]].to_numpy()# defines the features ex[50,10], [53,12]
                                                        # each cell trace over many T
            # TODO center
            start_x = x[0]
            x_centered = x -start_x
            start_y = y[0]
            y_centered = y - start_y
            # TODO call `add_features`
            features = np.column_stack([x_centered, y_centered])# stack the new xy values 
                                                                #so they go back into a list of lists
            features = add_features([features], features_list=features_list)[0]
            print(features[:5])                                                    

            track = Track( # Use the class track to get the individual track objects
                start_x=x[0],
                start_y=y[0],
                timepoints=timepoints,
                track_id=track_id,
                features=features.astype(float),
            )

            tracks.append(track)

        return tracks


X_data_struct = load_X_modified(datapath, filename_X, features_list=['XYZ', 'SL', 'DP'])
X_to_eval = [x.features for x in X_data_struct]

# %%
X_to_reference = load_X(datapath, filename_X, features=['XYZ', 'SL', 'DP'])

for my_x, orig_x in zip(X_to_eval, X_to_reference):
     assert np.allclose(my_x, orig_x)

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

# make a helper function that transofmrs the results back to the og coordinate positions
def recenter_coords(track):
    
    # TODO uncenter
    # get centered x and y from track.features

    # get the startx and the satrt y from the class track
    start_x = track.start_x
    start_y = track.start_y
    features = track.features
    track_id = track.track_id
    timepoints = track.timepoints

    # make empoty lists to add the new transofmed x and y values to 
    # x_uncentered_list = []
    # y_uncentered_list = []

    # # append new values to the bew x and y uncentered lists
    # for x_centered, y_centered in features[:, :2]:
    #     x_uncentered = x_centered + start_x
    #     y_uncentered = y_centered +  start_y
    #     x_uncentered_list.append(x_uncentered)
    #     y_uncentered_list.append(y_uncentered)

    #     features_uncentered = np.column_stack([x_uncentered_list, y_uncentered_list, track_id, timepoints])
    # index to get x and y uncentered into a list
    xy_uncentered = features[:, :2] + [[start_x, start_y]]  # (N, 2)

    # take the id and make it as long as the whole trac (ie, as long as the track id)
    track_id_col = np.full(len(timepoints), track_id)# (shape, and then value)

    output_content = np.column_stack([track_id_col, 
                                      timepoints,  
                                      xy_uncentered[:, 1],  # get only y
                                      xy_uncentered[:, 0] ]) # get only x


    output = pd.DataFrame(output_content, columns=["track_id", "t", "y", "x"]
)
    return output






                #     start_x=x[0],
                # start_y=y[0],
                # timepoints=timepoints,
                # track_id=track_id,
                # features=features.astype(float),

    return features_uncentered
    
# %%

# Uncenter the data by ading the original x values back to each vector
uncentered_test = recenter_coords(X_data_struct[0])
print(uncentered_test)

# %%
data = []

for i in range(len(predictions)):
    x = recenter_coords(X_data_struct[i])
    pred = predictions[i]
    x["pred"] = pred
    # data.append(pd.DataFrame({
    #     "track_id": X_data_struct[i].track_id,
    #     "y": x[:, 1],
    #     "x": x[:, 0],
    #     "t": X_data_struct[i].time_points,
    #     "pred": pred,
    # }))
    data.append(x)

df = pd.concat(data)
# %%
df.to_csv("predictions.csv", index=False)
# %%


    # %%
