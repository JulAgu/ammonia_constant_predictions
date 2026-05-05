import torch
import random
import pandas as pd
import numpy as np




def generate_tensors_response(df, pmid, response, device):
    
    data_filtered = df[df['pmid'] == pmid]
    
    response_tensor = torch.tensor(data_filtered[response].values, dtype=torch.float32).unsqueeze (1)
    
    return response_tensor.to(device)




def generate_tensors_predictors(df, pmid, cont_vars, cat_vars, device):
    
    data_filtered = df[df['pmid'] == pmid]

    x_cont = data_filtered[cont_vars]

    x_cont_tensor = torch.tensor(x_cont.values, dtype=torch.float32).view(len(x_cont), len(x_cont.columns))
    x_cont_tensor = x_cont_tensor.to(device)
    
    x_cat = data_filtered[cat_vars]
    
    x_cat_tensor = torch.tensor(x_cat.values, dtype=torch.long).view(len(x_cat), len(x_cat.columns))
    x_cat_tensor = x_cat_tensor.to(device)
    x_cat_tensor = torch.unbind (x_cat_tensor, dim = 1)

    output = [x_cont_tensor, x_cat_tensor]
    
    return output




def make_tensors (pmids_train, data, response, cont_vars, cat_vars, DEVICE):
        
    x_train = []
    y_train = []
    
    for pmid in pmids_train:
        
        x = generate_tensors_predictors(data, pmid, cont_vars, cat_vars, DEVICE)
        x_train.append (x)   
    
        y = generate_tensors_response(data, pmid, response, DEVICE)
        y_train.append (y)   
    
    return (x_train, y_train)




def make_mini_batches (x_train, y_train, batch_size):
    
    data_pairs = list(zip(x_train, y_train))
   
    random.shuffle(data_pairs)
    
    x_tensors_train_shuffled, y_tensors_train_shuffled = zip(*data_pairs)
    
    mini_batches_x = [x_tensors_train_shuffled[i:i + batch_size] for i in range(0, len(x_tensors_train_shuffled), batch_size)]
    mini_batches_y = [y_tensors_train_shuffled[i:i + batch_size] for i in range(0, len(y_tensors_train_shuffled), batch_size)]

    return (mini_batches_x, mini_batches_y)  




def train_model (model, num_epochs, learning_rate, x_train, y_train, DEVICE, batch_size = 32):
       
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    model.train()  

    list_out_fc1_before_relu = []
    list_out_fc1_after_relu = []
    list_out_fc2_before_relu = []
    list_out_fc2_after_relu = []
    
    for epoch in range(num_epochs):
        
        mini_batches_x, mini_batches_y = make_mini_batches (x_train, y_train, batch_size)
        all_predictions = []
        all_target_train = []

        n_mini_batches = len (mini_batches_x)

        for i in range(n_mini_batches):

            outputs, packed_y = model(mini_batches_x[i], mini_batches_y[i])  
            
            loss = torch.mean ((outputs - packed_y[0]) ** 2)    

            with torch.no_grad():

                _, out_fc1_before_relu, out_fc1_after_relu, out_fc2_before_relu, out_fc2_after_relu = model ([x_train[0]])
                    
            list_out_fc1_before_relu.append (out_fc1_before_relu.item())
            list_out_fc1_after_relu.append (out_fc1_after_relu.item())
            list_out_fc2_before_relu.append (out_fc2_before_relu.item())
            list_out_fc2_after_relu.append (out_fc2_after_relu.item())
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return (list_out_fc1_before_relu, list_out_fc1_after_relu, list_out_fc2_before_relu, list_out_fc2_after_relu)




def predict_emissions (data_origin, model, pmids, cont_vars, cat_vars, response, DEVICE):

    data_predictions = data_origin.copy()

    data_predictions = data_predictions[data_predictions['pmid'].isin (pmids)]
    pmids = data_predictions['pmid'].unique().tolist() # reordering the pmids to match the order in data_predictions

    data_predictions['prediction_ecum'] = None
        
    with torch.no_grad():
    
        all_predictions = torch.empty(0).to(DEVICE)
        
        preds = []
        
        for i in pmids:
    
            x = generate_tensors_predictors (data_predictions, i, cont_vars, cat_vars, DEVICE)
            y, *_ = model([x])
            preds.append (y.squeeze (-1))
            
        all_predictions = torch.cat (preds, dim = 0)
        data_predictions['prediction_ecum'] = all_predictions.detach().cpu().numpy()
        data_predictions['prediction_ecum'] = data_predictions['prediction_ecum'] * data_origin['e_cum_origin'].abs().max()
    
    return data_predictions
