import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from itertools import combinations, chain
from scipy.special import comb


def comb_index(n, k):
    count = comb(n, k, exact=True)
    index = np.fromiter(chain.from_iterable(combinations(range(n), k)), 
                        int, count=count*k)
    return index.reshape(-1, k)

def bin_feature(X, bins, feature_name):
    if len(feature_name) == 0:
        df = pd.DataFrame(X)
    else:
        df = pd.DataFrame(X, columns = feature_name)
    features = list(df.columns.values)
    for f in features:
        df[f] = pd.cut(df[f], bins, labels=False)
    return df

def duo_model(m_norm, m_imp, X, bins, feature_name):
    print("Creating Feature Map")
    feature_map = {"norm_imp": [], "norm_unimp": [], "abnorm_imp": [], "abnorm_unimp": []}
    for i in range(len(X)):
        if m_norm[i] == -1 and m_imp[i] == 1:
            feature_map['abnorm_unimp'] += [i]
        elif m_norm[i] == -1 and m_imp[i] == -1:
            feature_map['abnorm_imp'] += [i]
        elif m_norm[i] == 1 and m_imp[i] == 1:
            feature_map['norm_unimp'] += [i]
        elif m_norm[i] == 1 and m_imp[i] == -1:
            feature_map['norm_imp'] += [i]


    X = bin_feature(X, bins, feature_name)
    features = list(X.columns.values)

    shared_feature_map = {}
    shared_feature_map_name = {}
    key_cnts = []
    for key, value in tqdm(feature_map.items()):
        key_cnts.append((key, len(value)))
        common_cnt = np.zeros(len(features))
        common_cnt_detailed = np.full(len(features),np.nan)
        if len(value) == 0:
            shared_feature_map[key] = common_cnt
            shared_feature_map_name[key] = common_cnt_detailed
        else:
            temp_df = X.iloc[value,:]
            for i,f in enumerate(features):
                common_cnt[i] = temp_df[f].value_counts().iloc[0]
                common_cnt_detailed[i] = temp_df[f].value_counts().index[0]
            shared_feature_map[key] = common_cnt
            shared_feature_map_name[key] = common_cnt_detailed
    
    shared_feature_map = pd.DataFrame.from_dict(shared_feature_map, orient='index', columns = features)
    shared_feature_map_name = pd.DataFrame.from_dict(shared_feature_map_name, orient='index', columns = features)
    
    return X, shared_feature_map, shared_feature_map_name, feature_map, key_cnts

def analyze_shared_feature(f_max, data_binned, common_bins, feature_map_keys):
    f_comb_df = pd.DataFrame(columns=["abnorm_norm_imp", "abnorm_norm_unimp", "abnorm_imp_unimp", "norm_imp_unimp"])
    f_comb_df_idx = pd.DataFrame(columns=["abnorm_norm_imp", "abnorm_norm_unimp", "abnorm_imp_unimp", "norm_imp_unimp"])

    for key, value in tqdm(feature_map_keys.items()):
        if len(value) == 0:
            continue
        #* abnorm_unimp <-> norm_unimp
        if key == 'abnorm_unimp': 
            f_norm = common_bins.loc["norm_unimp"].to_numpy()
            k = 1

            for idx in value:
                empty = [0]*len(feature_map_keys.keys())
                empty_idx = [[]]*len(feature_map_keys.keys())
                instance = data_binned.iloc[idx].to_numpy()
                dif_f = np.where(f_norm != instance)[0]

                if len(dif_f) <= f_max:
                    dif_f = ','.join(map(str, dif_f))
                    if dif_f in f_comb_df.index:
                        f_comb_df.loc[dif_f, "abnorm_norm_unimp"] += 1

                        h = list(f_comb_df_idx.at[dif_f, "abnorm_norm_unimp"])
                        h.append(idx)
                        f_comb_df_idx.at[dif_f, "abnorm_norm_unimp"] = h
                    else:
                        empty[k] = 1
                        f_comb_df.loc[dif_f] = empty

                        empty_idx[k] = [idx]
                        f_comb_df_idx.loc[dif_f] = empty_idx
    
        #! * abnorm_imp <-> abnorm_unimp
        if key == 'abnorm_imp':
            f_norm = common_bins.loc["abnorm_unimp"].to_numpy()
            k = 2

            for idx in value:
                empty = [0]*len(feature_map_keys.keys())
                empty_idx = [[]]*len(feature_map_keys.keys())
                instance = data_binned.iloc[idx].to_numpy()
                dif_f = np.where(f_norm != instance)[0]
                if len(dif_f) <= f_max:
                    dif_f = ','.join(map(str, dif_f))
                    if dif_f in f_comb_df.index:
                        f_comb_df.loc[dif_f, "abnorm_imp_unimp"] += 1

                        h = list(f_comb_df_idx.at[dif_f, "abnorm_imp_unimp"])
                        h.append(idx)
                        f_comb_df_idx.at[dif_f, "abnorm_imp_unimp"] = h
                    else:
                        empty[k] = 1
                        f_comb_df.loc[dif_f] = empty

                        empty_idx[k] = [idx]
                        f_comb_df_idx.loc[dif_f] = empty_idx
    

        #! * norm_imp <-> norm_unimp
        if key == 'norm_imp':
            f_norm = common_bins.loc["norm_unimp"].to_numpy()
            k = 3

            for idx in value:
                empty = [0]*len(feature_map_keys.keys())
                empty_idx = [[]]*len(feature_map_keys.keys())
                instance = data_binned.iloc[idx].to_numpy()
                dif_f = np.where(f_norm != instance)[0]
                if len(dif_f) <= f_max:
                    dif_f = ','.join(map(str, dif_f))
                    if dif_f in f_comb_df.index:
                        f_comb_df.loc[dif_f, "norm_imp_unimp"] += 1

                        h = list(f_comb_df_idx.at[dif_f, "norm_imp_unimp"])
                        h.append(idx)
                        f_comb_df_idx.at[dif_f, "norm_imp_unimp"] = h
                   
                    else:
                        empty[k] = 1
                        f_comb_df.loc[dif_f] = empty

                        empty_idx[k] = [idx]
                        f_comb_df_idx.loc[dif_f] = empty_idx
    

        #* abnorm_imp <-> norm_imp
        if key == 'abnorm_imp':
            f_norm = common_bins.loc["norm_imp"].to_numpy()
            k = 0

            for idx in value:
                empty = [0]*len(feature_map_keys.keys())
                empty_idx = [[]]*len(feature_map_keys.keys())
                instance = data_binned.iloc[idx].to_numpy()
                dif_f = np.where(f_norm != instance)[0]
                if len(dif_f) <= f_max:
                    dif_f = ','.join(map(str, dif_f))
                    if dif_f in f_comb_df.index:
                        f_comb_df.loc[dif_f, "abnorm_norm_imp"] += 1
                        
                        h = list(f_comb_df_idx.at[dif_f, "abnorm_norm_imp"])
                        h.append(idx)
                        f_comb_df_idx.at[dif_f, "abnorm_norm_imp"] = h
                        
                    else:
                        empty[k] = 1
                        f_comb_df.loc[dif_f] = empty

                        empty_idx[k] = [idx]
                        f_comb_df_idx.loc[dif_f] = empty_idx
    

    return  f_comb_df,f_comb_df_idx

def get_key_color(key_name):
    cmap = matplotlib.cm.get_cmap('Accent')
    if key_name == "norm_imp":
        c = cmap(0)
    elif key_name == "norm_unimp":
        c = cmap(0.33)
    elif key_name == "abnorm_imp":
        c = cmap(0.67)
    else:
        c = cmap(0.99)
    return c

def plt_result(data_name, norm_df, key_name, f_names, key1, key2, day_cnt):
    f = norm_df.iloc[[key1, key2], 0:-1]
    y_len = len(f_names)//3
    if y_len < 5:
        y_len = 5
    plt.figure(figsize=(y_len, 4))
    sns.set_theme(style="whitegrid")
    a = f.iloc[[0]].values.tolist()[0]
    b = f.iloc[[1]].values.tolist()[0]

    abs_dif = []
    for i, val in enumerate(a):
        abs_dif.append(abs((val - b[i]))*-1)
    sorted_idx = np.argsort(abs_dif)

    f_names = [f_names[i] for i in sorted_idx] 
    if not isinstance(f_names[0], str):
        f_names = [str(i) for i in f_names]
    a = [a[i] for i in sorted_idx] 
    b = [b[i] for i in sorted_idx] 
    plt.title(data_name)
    plt.vlines(f_names, ymin=a, ymax=b, linewidth=2, color='k')
    plt.scatter(f_names, a, label=key_name[key1], alpha= 1, color = get_key_color(key_name[key1]))
    plt.scatter(f_names, b, label=key_name[key2], alpha= 1, color = get_key_color(key_name[key2]))
    plt.xlabel('Features')
    plt.xticks(rotation = 45)
    sns.despine(left=True)
    plt.legend()
    plt.show()


