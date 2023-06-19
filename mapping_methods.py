import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras import regularizers

from sklearn.metrics.pairwise import cosine_similarity
from sklearn import linear_model
from sklearn.decomposition import PCA, TruncatedSVD

import torch
from torch.nn.functional import normalize

import numpy as np
import scipy as sp
import sys
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import gc

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
print(torch.cuda.is_available())

model_plot_names = {
    "bert-base-multilingual-uncased": "mBERT uncased",
    "mt5-base": "MT5",
    "xlm-roberta-base": "XLM-Roberta",
    "ernie-m-base_pytorch": "ErnieM"    
}

def mate_retrieval(l1_vecs, l2_vecs):
    l1_vecs, l2_vecs = torch.as_tensor(l1_vecs).to(device), torch.as_tensor(l2_vecs).to(device)
    l1_vecs, l2_vecs = normalize(l1_vecs).to(device), normalize(l2_vecs).to(device)
    sim = torch.matmul(l1_vecs,l2_vecs.T).to(device)
    '''Mate retrieval rate - the rate when the most symmetric document is ones translation.'''
    aux = sum([sim[i].argmax()==i for i in range(sim.shape[0])])/sim.shape[0]
    aux = aux.cpu().detach().numpy().item(0)
    return aux

def rank(val,a):
    if sp.sparse.issparse(a):
        return a[a>=val].shape[1] #if a is sparse
    return len(a[a>=val]) # if a is dense

def reciprocal_rank(l1_vecs, l2_vecs):
    '''Mean reciprocal rank'''
    if torch.is_tensor(l1_vecs):
        l1_vecs, l2_vecs = l1_vecs.cpu().detach().numpy(), l2_vecs.cpu().detach().numpy()
        
    sim = cosine_similarity(l1_vecs, l2_vecs)

    return sum([1/rank(sim[i,i],sim[i]) for i in range(sim.shape[0])])/sim.shape[0]

def lca_mapping(source_train, source_test, target_train, target_test):
  lca = torch.matmul(torch.linalg.pinv(target_train.T),source_train.T).to(device)
  linear_mapping = lambda x: torch.matmul(lca.T, x).to(device)
  return linear_mapping(target_test)

def lca(sl_train, sl_test, tl_train, tl_test, dims, evaluation_function):
  last_map = []
  nmax = [-1,-1]
  nmax_dim = [-1,-1]

  mate_scores = []
    
  sl_no_mapping_all = torch.cat((sl_train,sl_test)).to(device)
  tl_no_mapping_all = torch.cat((tl_train,tl_test)).to(device)

  for dim in tqdm(dims, desc="LCA"):
    sl_train_r, sl_test_r = sl_train[:,:dim].T, sl_test[:,:dim].T
    tl_train_r, tl_test_r = tl_train[:,:dim].T, tl_test[:,:dim].T

    sl_no_mapping = sl_no_mapping_all[:,:dim]
    tl_no_mapping = tl_no_mapping_all[:,:dim]

    lm_sl_tl = lca_mapping(sl_train_r, sl_test_r, tl_train_r, tl_test_r)
    
    mr_sl_tl = evaluation_function(sl_test_r.T,lm_sl_tl.T)
    mr_no_mapping = evaluation_function(sl_no_mapping,tl_no_mapping)

    mate_scores.append([mr_sl_tl, mr_no_mapping, dim])

    if mr_sl_tl > nmax[0]:
      nmax[0] = mr_sl_tl
      nmax_dim[0] = dim
    if mr_no_mapping > nmax[1]:
      nmax[1] = mr_no_mapping
      nmax_dim[1] = dim 
        
  print(nmax, nmax_dim)

  return mate_scores

def clear_memory():
  tf.keras.backend.clear_session() # clear cache after every mapping method
  torch.cuda.empty_cache()
  torch.cuda.synchronize()
  print(torch.cuda.memory_summary())

def lcc(en_train_matrix,
                en_test_matrix,
                fr_train_matrix,
                fr_test_matrix,
                dimensions,
                evaluation_function):

    scores = []
    
    nmax = [-1,-1]
    nmax_dim = [-1,-1]
    
    sl_no_mapping_all = torch.cat((en_train_matrix,en_test_matrix)).to(device)
    tl_no_mapping_all = torch.cat((fr_train_matrix,fr_test_matrix)).to(device)
    
    for dimension in tqdm(dimensions, desc="LCC"):
        en = en_train_matrix[: ,:dimension] - torch.mean(en_train_matrix[:,:dimension], axis=0).to(device)
        fr = fr_train_matrix[: ,:dimension] - torch.mean(fr_train_matrix[:,:dimension], axis=0).to(device)
        
        sl_no_mapping = sl_no_mapping_all[:,:dimension]
        tl_no_mapping = tl_no_mapping_all[:,:dimension]
        
        sample_size = en.shape[0]
        zero_matrix = torch.zeros((sample_size, dimension)).to(device)
        X1 = torch.cat((en, zero_matrix), axis = 1).to(device)
        X2 = torch.cat((zero_matrix, fr), axis= 1).to(device)
        X = torch.cat((X1, X2), axis = 0).to(device)
        Y1 = torch.cat((en, fr), axis = 1).to(device)
        Y2 = torch.cat((en, fr), axis = 1).to(device)
        Y = torch.cat((Y1, Y2), axis = 0).to(device)

        X = X.cpu().detach().numpy()
        Y = Y.cpu().detach().numpy()
        
        reg = linear_model.RidgeCV(alphas=[1e-10, 1e-3, 1e-2, 1e-1, 1, 10])
        reg.fit(X,Y)
        pca = PCA(n_components = int(dimension))
        pca.fit(reg.predict(X))
        
        rrr = lambda X: np.matmul(pca.transform(reg.predict(X)), pca.components_)

        en = en_test_matrix[: ,:dimension] - torch.mean(en_train_matrix[:,:dimension], axis=0).to(device)
        fr = fr_test_matrix[: ,:dimension] - torch.mean(fr_train_matrix[:,:dimension], axis=0).to(device)
        zero_matrix = torch.zeros((en_test_matrix.shape[0], dimension)).to(device)
        X1 = torch.cat((en, zero_matrix), axis = 1).to(device)
        X2 = torch.cat((zero_matrix, fr), axis= 1).to(device)
        X = torch.cat((X1, X2), axis = 0).to(device)
        
        X1 = X1.cpu().detach().numpy()
        X2 = X2.cpu().detach().numpy()
        
        english_encodings_lcc = rrr(X1)
        french_encodings_lcc = rrr(X2)
        score_sl_tl = evaluation_function(english_encodings_lcc, french_encodings_lcc)
        score_no_mapping = evaluation_function(sl_no_mapping, tl_no_mapping)
        
        scores.append([score_sl_tl, score_no_mapping, dimension])
        
        if score_sl_tl > nmax[0]:
            nmax[0] = score_sl_tl
            nmax_dim[0] = dimension
        if score_no_mapping > nmax[1]:
            nmax[1] = score_no_mapping
            nmax_dim[1] = dimension 

    print(nmax, nmax_dim)

    return scores


def nnca(l1_train, l1_test, l2_train, l2_test, dimensions, settings, evaluation_function, is_plotting=False): 
  neurons = settings["neurons"]
  activation_function = settings["activation_function"]
  max_epochs = settings["max_epochs"]
  dropout = settings["dropout"]
  loss = settings["loss_function"]
  learning_rate = settings["learning_rate"]

  callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
  scores = []
    
  sl_no_mapping_all = torch.cat((l1_train,l1_test)).to(device)
  tl_no_mapping_all = torch.cat((l2_train,l2_test)).to(device)
    
  l1_train, l1_test = tf.convert_to_tensor(l1_train.cpu().detach().numpy()), tf.convert_to_tensor(l1_test.cpu().detach().numpy())
  l2_train, l2_test = tf.convert_to_tensor(l2_train.cpu().detach().numpy()), tf.convert_to_tensor(l2_test.cpu().detach().numpy())  

  for idz, dimension in tqdm(enumerate(dimensions), desc = "NNCA"):
    x = tf.keras.layers.Input(shape=(dimension,))
    d1 = tf.keras.layers.Dropout(dropout)(x)
    
    l = tf.keras.layers.Dense(neurons[0], activation = activation_function)(d1)
    for no_neurons in neurons[1:]:
        l = tf.keras.layers.Dense(no_neurons, activation = activation_function)(l)
        
    d2 = tf.keras.layers.Dropout(dropout)(l)
    y = tf.keras.layers.Dense(dimension, activation = None)(d2)
    
    if loss == "cosine_sim":
      loss = tf.keras.losses.CosineSimilarity(axis=0)

    l1_to_l2_clf = Model(x, y)
    l2_to_l1_clf = Model(x, y)

    l1_to_l2_clf.summary()

    l1_to_l2_clf.compile(optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss = loss,
            metrics= tf.keras.losses.CosineSimilarity()
            )

    l2_to_l1_clf.compile(optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss = loss,
            metrics= tf.keras.losses.CosineSimilarity())


    hist1 = l1_to_l2_clf.fit(l1_train[: ,:dimension], 
                      l2_train[:,:dimension], 
                      epochs=max_epochs, 
                      validation_data = (l1_test[: ,:dimension], l2_test[: ,:dimension]), 
                      callbacks=[callback], verbose=0)

    hist2 = l2_to_l1_clf.fit(l2_train[: ,:dimension], 
                      l1_train[: ,:dimension],
                      epochs=max_epochs, 
                      validation_data = (l2_test[: ,:dimension], l1_test[: ,:dimension]), 
                      callbacks=[callback], verbose=0)  

    fake_fr = l1_to_l2_clf.predict(l1_test[: ,:dimension])
    fake_en = l2_to_l1_clf.predict(l2_test[: ,:dimension])

    merged_trans_vecs = np.concatenate((fake_en, l2_test[:,:dimension]), axis = 1)
    real_vecs = np.concatenate((l1_test[:,:dimension], fake_fr), axis = 1)
    
    sl_no_mapping = sl_no_mapping_all[:,:dimension]
    tl_no_mapping = tl_no_mapping_all[:,:dimension]

    score = evaluation_function(merged_trans_vecs, real_vecs)
    score_no_mapping = evaluation_function(sl_no_mapping, tl_no_mapping)
   
    scores.append([score, score_no_mapping, dimension])
    
    print(f"{dimension} - {score} - {score_no_mapping}")
    
    # if is_plotting:
    #     add_lineplot(hist1.history["loss"], f"loss - sl-tl", "min", "NNCA", "Loss")
    #     add_lineplot(hist1.history["val_loss"], f"val_loss - sl-tl", "min", "NNCA", "Loss")
    #     plt.show()
    #     add_lineplot(hist1.history["cosine_similarity"], f"cosine_similarity - sl-tl", "min", "NNCA", "cosine_similarity")
    #     add_lineplot(hist1.history["val_cosine_similarity"], f"val_cosine_similarity - sl-tl", "min", "NNCA", "cosine_similarity")
    #     plt.show()

    
    del l1_to_l2_clf
    del l2_to_l1_clf
    
    del hist1, hist2, fake_fr, fake_en, merged_trans_vecs, real_vecs, score, score_no_mapping
  
  return scores

def add_lineplot(data, name, minmax, method, metric):
    sns.set_theme(style="whitegrid")
    sns.color_palette("husl")
    sns.despine()
    
    if minmax == "min":
        point_x = min
        point_y = np.argmin
    if minmax == "max":
        point_x = max
        point_y = np.argmax

    max_point = point_x(data)

    plot = sns.lineplot(x=range(len(data)), y=data, label=name + " " + method)
    sns.scatterplot(x=[point_y(data)], y=[max_point], marker="x", s=100, label=f"{name} - {max_point:.3f}", ax=plot)

    plt.title(f"{metric} across epochs", fontsize=50)
    plt.xlabel("Epoch", fontsize=30)
    plt.xticks(fontsize=25)
    plt.yticks(fontsize=25)
    plt.ylabel(metric, fontsize=30)
    sns.move_legend(plot, "upper left", bbox_to_anchor=(1,1), fontsize=20)

def plot_mate_scores(sl, tl, mate_scores, models, dims, method, path, evaluation_function, display=False):
  sns.set_theme(style="whitegrid")
  sns.color_palette("husl")
  sns.despine()
  plt.figure(figsize=(16,9))
    
  plot = None

  sl, tl = sl.upper(), tl.upper()
  evaluation_function = evaluation_function.title()
  evaluation_function = " ".join(evaluation_function.split("_"))
    
  aux = []
  for model in models:
    if model in model_plot_names.keys():
        model_name = model_plot_names[model]
    else:
        model_name = model
    aux.append(model_name)
    
  models = aux

  for idx, score in enumerate(mate_scores):
    score = np.array(score)
    
    y_en = score[:,0]
    y_no_mapping = score[:,1]
    y_en_max = np.argmax(y_en)
    y_no_mapping_max = np.argmax(y_no_mapping)
    x_dim = dims

    plot = sns.lineplot(x=x_dim,y=y_en,label=f"{models[idx]}")
    sns.scatterplot(y=[y_en[y_en_max]], x=[x_dim[y_en_max]],label=f"{x_dim[y_en_max]} - {y_en[y_en_max]:.4f}", marker="^",s=100)
    
    sns.lineplot(x=x_dim,y=y_no_mapping,label=f"{models[idx]} - no mapping")
    sns.scatterplot(y=[y_no_mapping[y_no_mapping_max]], x=[x_dim[y_no_mapping_max]],label=f"{x_dim[y_no_mapping_max]} - {y_no_mapping[y_no_mapping_max]:.4f}", marker="^",s=100)
  
  plt.title(f"{evaluation_function} across dimensions without mapping and using {method.upper()} for {sl}-{tl}", fontsize=40)
  
  plt.xlabel("Dimension", fontsize=30)
  plt.ylabel(evaluation_function.title(), fontsize=30)
  plt.xticks(fontsize=25)
  plt.yticks(fontsize=25)
  sns.move_legend(plot, "upper left", bbox_to_anchor=(1,1), fontsize=20)

  plt.savefig(path + f"{'.'.join(models)}_{method}_{sl}_{tl}.jpg", bbox_inches="tight")
  plt.show() if display else 1