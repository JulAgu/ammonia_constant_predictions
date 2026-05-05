import torch
import seaborn as sns
import pandas as pd
import numpy as np
import random
import importlib

import sys
sys.path.append ('functions')

import utils as mf
importlib.reload(mf) 

from rnn_module import AmmoniaRNN

DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'

data = pd.read_csv("data/data.csv")

data = data.rename(columns={"id": "pmid"})
data = data.drop(['Unnamed: 0'], axis = 1)

data_origin = data.copy() # pour la dénormalisation dans la fonction predict_emissions

pmids = data['pmid'].unique().tolist()

random.seed (1)
pmids_train = random.sample (pmids, 500)


# model parameters -------------------------------------------------------------------------
cat_vars_full = [
    'app_method', 'incorp', 'till',
    'meas_tech', 
    'fer_origin', 'fer_forme', 'fer_trt1', 'fer_trt2', 'fer_trt3',
    'crop',
    'soil_type'
]

cont_vars_full = [
    'ct', 'dt', 
    'air_temp', 'wind_2m', 'rain_rate', 'rain_cum',
    'tan_app', 'app_rate', 
    'fer_dm', 'fer_ph', 
    'time_incorp', 'furrow_z',
    'soil_water', 'soil_ph', 'soil_dens', 'soil_oc',
    'crop_z',
    
    'air_temp_ind', 'wind_2m_ind',
    'fer_dm_ind', 'fer_ph_ind', 
    'soil_water_ind', 'soil_ph_ind', 'soil_dens_ind', 'soil_oc_ind', 
    'crop_z_ind',
    'time_incorp_ind'
]

num_layers = 1
nonlinearity = "relu"
bidirectional = True
response = "e_cum"
hidden_size = 512 

output_size = 1

cont_vars = [x for x in cont_vars_full if x != "soil_oc"]
cat_vars = cat_vars_full

cat_dims = [max (data[x]) + 1 for x in cat_vars]  
embedding_dims = [8] * len (cat_vars)  

input_size = len (cont_vars) + len (cat_vars)
# ------------------------------------------------------------------------------------------




# training ---------------------------------------------------------------------------------
l = []
list_predictions = []

for seed in range (5):
    random.seed (seed)
    print (f"seed = {seed}")
    

    torch.manual_seed(1)
    model = AmmoniaRNN(
        input_size = input_size, 
        output_size = output_size, 
        hidden_size = hidden_size, 
        nonlinearity = nonlinearity,
        num_layers = num_layers,
        bidirectional = bidirectional,
        cat_dims = cat_dims, 
        embedding_dims = embedding_dims
    ).to(DEVICE)
    
    x_train, y_train = mf.make_tensors (pmids_train, data, response, cont_vars, cat_vars, DEVICE)
    
    num_epochs = 5
    learning_rate = 5e-4
    list_out_fc1_before_relu, list_out_fc1_after_relu, list_out_fc2_before_relu, list_out_fc2_after_relu = mf.train_model (model, num_epochs, learning_rate, x_train, y_train, DEVICE)

    df_tmp = pd.concat ([
        pd.DataFrame ({"position" : "sum (abs (h)) fc1 before relu", "value" : list_out_fc1_before_relu, "seed": seed}).reset_index(),
        pd.DataFrame ({"position" : "sum (abs (h)) fc1 after relu", "value" : list_out_fc1_after_relu, "seed": seed}).reset_index(),
        pd.DataFrame ({"position" : "sum (abs (h)) fc2 before relu", "value" : list_out_fc2_before_relu, "seed": seed}).reset_index(),
        pd.DataFrame ({"position" : "sum (abs (h)) fc2 after relu", "value" : list_out_fc2_after_relu, "seed": seed}).reset_index()
    ])

    l.append (df_tmp)

    data_predictions = mf.predict_emissions (data_origin, model, pmids_train, cont_vars, cat_vars, response, DEVICE)

    data_predictions = data_predictions.loc[data_predictions.groupby(['pmid'])['ct'].idxmax()]

    data_predictions = data_predictions.assign (seed = seed)
    list_predictions.append (data_predictions)
# ------------------------------------------------------------------------------------------


# plots ------------------------------------------------------------------------------------
df_plot1 = pd.concat (list_predictions)

g = sns.FacetGrid (df_plot1, col = "seed")
g.map_dataframe (sns.scatterplot, x = "e_cum_origin", y = "prediction_ecum")

g.savefig("results/obs_vs_pred_less_variables.png", dpi=200, bbox_inches="tight")

df_plot2 = pd.concat (l)

g = sns.FacetGrid (df_plot2, col = "seed", hue = "position", col_wrap = 5)
g.map_dataframe (sns.scatterplot, x = "index", y = "value")
g.add_legend()

g.savefig("results/sum_abs_h_less_variables.png", dpi=200, bbox_inches="tight")
# ------------------------------------------------------------------------------------------
