import json
import keras
import os
import pandas as pd
import time
import torch
import multiprocessing as mp

from itertools import permutations
from mapping_methods import *
from sklearn.model_selection import train_test_split

PATH = "Thesis/Embeddings"
CORES = 20

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
print(torch.cuda.is_available())
        
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

map_dict = {
    "lca": lca,
    "lcc": lcc,
    "nnca": nnca
}

settings = settings_nnca[0]

languages = ["en", "ro", "fr", "de", "nl"]
language_pairs = list(permutations(languages,2))
models = ["bert-base-multilingual-uncased", "mt5-base", "xlm-roberta-base", "ernie-m-base_pytorch"]
size = 7000
dims = [100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 768]
mappings = [map_dict[sys.argv[1]]]
evaluation_functions = [reciprocal_rank, mate_retrieval]

if sys.argv[1] == "nnca": # TF is allocating all GPU memory if there is more than 1 process, I don't know at the moment how to deal with that
    CORES = 1

### DEBUGGING 
# dims = [250, 450, 768]
# evaluation_functions = [reciprocal_rank]
# size = 50
# language_pairs = list(permutations(languages,2))[:4]
# mappings = [lca, nnca]

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
                
    return score

def evaluate_mapping(sl, tl, size, dims, evaluation_function, path_plot, path_result):
    language_pair = f"{sl}-{tl}"
    start_time = time.time()
    for mapping in mappings:
        if f"{mapping.__name__}_{language_pair}.json" in os.listdir(path_result):
            print(f"{language_pair} for {mapping.__name__} is already present, moving to the next pair")
            continue

        print(language_pair, mapping.__name__, path_result)

        score_pair = {}
        for model in models:
            mate_scores = evaluate_method(sl, tl, model, size, mapping, dims, evaluation_function)
            score_pair[model] = mate_scores
        plot_mate_scores(sl, tl, [value for _, value in score_pair.items()], models, dims, mapping.__name__, path_plot, evaluation_function.__name__)
        
        with open(f"./Thesis/Results/{evaluation_function.__name__}/{language_pair}/{mapping.__name__}_{language_pair}.json", "w") as f:
            json.dump({language_pair: score_pair}, f)
            del score_pair
        
        clear_memory() if torch.cuda.is_available() else 1
    print(torch.cuda.memory_summary()) if torch.cuda.is_available() else 1            			
    print(f"It took {time.time() - start_time}s to process the data for {language_pair}")
    
if __name__ == "__main__":
    os.mkdir("Thesis/Plots") if "Plots" not in os.listdir("Thesis") else 1
    os.mkdir("Thesis/Results") if "Results" not in os.listdir("Thesis") else 1

    print(f"--------- Using {mappings[0].__name__}")

    mp.set_start_method('spawn')

    with torch.no_grad():
        for evaluation_function in evaluation_functions:
            start_time = time.time()
                    
            os.mkdir(f"Thesis/Plots/{evaluation_function.__name__}") if evaluation_function.__name__ not in os.listdir("Thesis/Plots") else 1
            os.mkdir(f"Thesis/Results/{evaluation_function.__name__}") if evaluation_function.__name__ not in os.listdir("Thesis/Results") else 1

            no_processes = 0
            processes = []
            for (sl, tl) in language_pairs:
                language_pair = f"{sl}-{tl}"
                
                path_plot = f"Thesis/Plots/{evaluation_function.__name__}/{language_pair}/"
                path_result = f"Thesis/Results/{evaluation_function.__name__}/{language_pair}/"

                os.mkdir(path_plot) if language_pair not in os.listdir(f"Thesis/Plots/{evaluation_function.__name__}") else 1
                os.mkdir(path_result) if language_pair not in os.listdir(f"Thesis/Results/{evaluation_function.__name__}") else 1

                p = mp.Process(target=evaluate_mapping, args=(sl, tl, size, dims, evaluation_function, path_plot, path_result))
                p.start()
                processes.append(p)
                no_processes += 1

                if no_processes >= CORES:
                    print(f"Reached {CORES} processes - joining")
                    for p in processes:
                        p.join()

                    no_processes = 0
                    processes = []
            
            if len(processes) > 0: # any remaining processes
                print(f"Joining the remaining {len(processes)} processes")
                for p in processes:
                        p.join()