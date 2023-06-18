import json
import keras
import os
import pandas as pd
import time
import torch

from itertools import permutations
from mapping_methods import *
from sklearn.model_selection import train_test_split

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
print(torch.cuda.is_available())

PATH = "Thesis/Embeddings"

print(os.listdir())

settings_nnca = [
    {
        "neurons": [500],
        "activation_function": "elu",
        "max_epochs": 250,
        "dropout": 0.0,
        "loss_function": "Huber",
        "learning_rate": 5e-4
    }
]

settings = settings_nnca[0]

def evaluate_method_transformer(sl, tl, model_name, size, method, dims, evaluation_function):
    sl_vec = np.load(PATH + "/" + f"{sl}/" + f"emb_{model_name}_{sl}.npy",mmap_mode="r")
    tl_vec = np.load(PATH + "/" + f"{tl}/" + f"emb_{model_name}_{tl}.npy",mmap_mode="r")

    sl_vec = torch.as_tensor(sl_vec).to(device)
    tl_vec = torch.as_tensor(tl_vec).to(device)

    print(f"SL data shape: {np.shape(sl_vec)} | TL data shape: {np.shape(tl_vec)}")

    sl_train, sl_test, tl_train, tl_test = train_test_split(sl_vec[:size], tl_vec[:size], test_size=0.25, random_state=42)

    print(f"Model: {model_name} | Training data shape: {np.shape(sl_train)} | Testing data shape: {np.shape(sl_test)}")

    if method == nnca:
        score = method(sl_train, sl_test, tl_train, tl_test, dims, settings, evaluation_function, is_plotting=False)
    else:    
        score = method(sl_train, sl_test, tl_train, tl_test, dims, evaluation_function)

    return score

def evaluate_method(sl, tl, model_name, size, method, dims, evaluation_function):
    score = []
    
    if model_name == "doc2vec":
        score = evaluate_method_doc2vec(sl, tl, model_name, size, method, dims, evaluation_function)
    else: # all transformer models
        score = evaluate_method_transformer(sl, tl, model_name, size, method, dims, evaluation_function)
        
    clear_memory() if torch.cuda.is_available() else 1
        
    return score


languages = ["en", "ro", "fr", "de", "nl", "es"][:5]

language_pairs = list(permutations(languages,2))

models = ["bert-base-multilingual-uncased", "mt5-base", "xlm-roberta-base", "ernie-m-base_pytorch"]

size = 5000

# dims = [250, 450, 768]
dims = [100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 768]

mappings = [lca, lcc, nnca]

evaluation_functions = [mate_retrieval, reciprocal_rank]
# evaluation_functions = [reciprocal_rank]

os.mkdir("Thesis/Plots") if "Plots" not in os.listdir("Thesis") else 1
os.mkdir("Thesis/Results") if "Results" not in os.listdir("Thesis") else 1

with torch.no_grad():
	for evaluation_function in evaluation_functions:
		start_time = time.time()
                
		os.mkdir(f"Thesis/Plots/{evaluation_function.__name__}") if evaluation_function.__name__ not in os.listdir("Thesis/Plots") else 1
		os.mkdir(f"Thesis/Results/{evaluation_function.__name__}") if evaluation_function.__name__ not in os.listdir("Thesis/Results") else 1
        
		for (sl, tl) in language_pairs:
			language_pair_score = {}
			language_pair = f"{sl}-{tl}"
            
			path = f"Thesis/Plots/{evaluation_function.__name__}/{language_pair}/"
			os.mkdir(path) if language_pair not in os.listdir(f"Thesis/Plots/{evaluation_function.__name__}") else 1

			if f"{language_pair}.json" in os.listdir(f"Thesis/Results/{evaluation_function.__name__}"):
				print(f"{language_pair} already present - moving to next language pair")
				continue

			mapping_scores = {}
			for mapping in mappings:
				score_pair = {}
				for model in models:
					mate_scores = evaluate_method(sl, tl, model, size, mapping, dims, evaluation_function)
					score_pair[model] = mate_scores
				plot_mate_scores(sl, tl, [value for key, value in score_pair.items()], models, dims, mapping.__name__, path, evaluation_function.__name__)
				mapping_scores[mapping.__name__] = score_pair
                
			print(torch.cuda.memory_summary())            			
			print(f"It took {time.time() - start_time}s to process the data for {language_pair}")
			
			with open(f"./Thesis/Results/{evaluation_function.__name__}/{language_pair}.json", "w") as f:
				json.dump({language_pair: mapping_scores}, f)