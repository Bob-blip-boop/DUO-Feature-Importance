import pandas as pd
from numpy.random import default_rng
import random
from collections import Counter
from skmultiflow.data import AGRAWALGenerator
import matplotlib.pyplot as plt
from matplotlib import ticker
import seaborn as sns
from datetime import datetime
import itertools
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from alibi.explainers import PartialDependence, plot_pd, ALE
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
from scipy import stats
import time
from tqdm import tqdm
import multiprocessing
import sys



#* Random Data Generation
def generate_data(num_instances, toltal_n_features, f_range,s):
    random.seed(s)

    data = []
    y = []
    for i in range(num_instances):
        f_val =[]
        for j in range(toltal_n_features):
            f = random.randint(0, f_range)
            f_val.append(f)
     
        #* Add the instance to the dataset
        data.append(f_val)
    f_names = []
    for i in range(toltal_n_features):
        f_names.append("f_" + str(i))

    return data, f_names

def get_func_labels(x, fnc):
    y = []
    new_x = []

    #! Original function 1
    if fnc == 1:
        for i in x:
            if i[2] < 20 and (i[7] > 10):
                y.append(1)
            elif i[2] >= 40 and (i[7] > 20):
                y.append(1)
            else:
                y.append(0)

    #! Original function 2
    elif fnc == 2:
        for i in x:
            if (i[2] < 40) and (i[0] >= 50000) and (i[0] <= 100000):
                y.append(1)
            elif (i[2] >= 40) and (i[2] < 60) and (i[0] >= 75000) and (i[0] <= 125000):
                y.append(1)
            elif (i[2] >= 60) and (i[0] >= 25000) and (i[0] <= 75000):
                y.append(1)
            else:
                y.append(0)
    
    
    #! Original function 3
    elif fnc == 3:
        for i in x:
            if (i[2] < 40) and (i[3] >= 0) and (i[3] <= 1):
                y.append(1)
            elif (i[2] >= 40) and (i[2] < 60) and (i[3] >= 1) and (i[3] <= 3):
                y.append(1)
            elif (i[2] >= 60) and (i[3] >= 2) and (i[3] <= 4):
                y.append(1)
            else:
                y.append(0)

    #! Important "age" and 'hyears' Feature value Shift
    elif fnc == 11:
        for i in x:
            if i[2] < 40 and (i[7] > 5):
                y.append(1)
            elif i[2] >= 60 and (i[7] > 10):
                y.append(1)
            else:
                y.append(0)


    #! Important "salary" and Unimportant "elevel" Feature Shift
    elif fnc == 12:
        for i in x:
            if (i[2] < 40) and (i[0] >= 25000) and (i[0] <= 10000):
                y.append(1)
            elif (i[2] >= 40) and (i[2] < 60) and (i[0] >= 100000) and (i[0] <= 125000):
                y.append(1)
            elif (i[2] >= 60) and (i[0] >= 50000) and (i[0] <= 100000):
                y.append(1)
            else:
                y.append(0)
            
            new_i = []
            for j in range(len(i)):
                if j == 3 :
                    new_i.append(i[j]+10)
                else:
                    new_i.append(i[j])
            new_x.append(new_i)
        return new_x, y
    
    #! Unimportant "salary", "commission", "loan" Feature Shift
    elif fnc == 13:
        for i in x:
            if (i[2] < 40) and (i[3] >= 0) and (i[3] <= 1):
                y.append(1)
            elif (i[2] >= 40) and (i[2] < 60) and (i[3] >= 1) and (i[3] <= 3):
                y.append(1)
            elif (i[2] >= 60) and (i[3] >= 2) and (i[3] <= 4):
                y.append(1)
            else:
                y.append(0)

            new_i = []
            for j in range(len(i)):
                if j == 0 or j == 1 or j == 8:
                    new_i.append((i[j]+50000)*2)
                else:
                    new_i.append(i[j])
            new_x.append(new_i)
        return new_x, y

    return x, y

def generate_argawal(num_instances, drift_loc, w, s, fnc_org, fnc_drifted):
    gen =  AGRAWALGenerator(random_state = s)
    X, y = gen.next_sample(num_instances)

    org_x, org_y = get_func_labels(X, fnc_org)
    new_x, new_y = get_func_labels(X, fnc_drifted)

    drifted_x = []
    drifted_y = []

    for i in range(num_instances):
        prob = shift_prob(i, drift_loc, w)
        rand_num = np.random.rand()
        #print(rand_num, prob)
        if rand_num > prob:
            drifted_x.append(org_x[i])
            drifted_y.append(org_y[i])
        else:
            drifted_x.append(new_x[i])
            drifted_y.append(new_y[i])

    return drifted_x, drifted_y, gen.feature_names

def single_interact_y(rand_data, cor_n_features):
    y = []
    for d in rand_data:
        if all(val == 0 for val in d[:cor_n_features]):
            y.append(1)
        else:
            y.append(0)
    return y

def single_interact_y_drifted(instance, cor_n_features):

    if all(val == 3 for val in instance[:cor_n_features]):
        drift_label = 1
    else:
        drift_label = 0
    
    drifted_inst = instance
    for j in range(cor_n_features):
            f = random.randint(10, 13)
            drifted_inst[j+cor_n_features] = f

    return drifted_inst, drift_label

def two_interact_y(rand_data, cor_n_features):
    y = []
    for d in rand_data:
        if all(val == 0 for val in d[:cor_n_features]):
            y.append(1)
        elif all(val == 1 for val in d[1:cor_n_features+1]):
            y.append(1)
        else:
            y.append(0)
    return y

def two_interact_y_drifted(instance, cor_n_features):
    if all(val == 3 for val in instance[:cor_n_features]):
        drift_label = 1
    elif all(val == 2 for val in instance[1:cor_n_features+1]):
        drift_label = 1
    else:
        drift_label = 0

    drifted_inst = instance
    for j in range(cor_n_features):
            f = random.randint(10, 13)
            drifted_inst[j+cor_n_features+1] = f

    return drifted_inst, drift_label

def single_interact_y_new(rand_data, cor_n_features, drift_loc):
    class_ratio = 0.1
    y = []
    for d in rand_data:
        if all(val == 0 for val in d[:cor_n_features]):
            y.append(1)
        else:
            y.append(0)
    if sum(y[:drift_loc]) >= class_ratio*drift_loc:
        return rand_data, y

    new_cnt = int(class_ratio*drift_loc) - sum(y[:drift_loc])
    idx =  random.randint(0,drift_loc-1)
    while new_cnt != 0:
        d = rand_data[idx]
        if all(val == 0 for val in d[:cor_n_features]):
                idx = random.randint(0,drift_loc-1)
                continue 
        else:
            rand_data[idx] = [0]*cor_n_features + rand_data[idx][cor_n_features:]
            y[idx] = 1
            new_cnt-=1
            idx = random.randint(0,drift_loc-1)


    return rand_data, y

def single_interact_y_drifted_new(rand_data, cor_n_features, drift_loc):
    new_feat = 1
    class_ratio = 0.1
    y = []
    for d in rand_data:
        if all(val == new_feat for val in d[:cor_n_features]):
            y.append(1)
        else:
            y.append(0)
    
    for d in range(len(rand_data)):
        if cor_n_features <= 100:
            rand_data[d] = rand_data[d][:cor_n_features] + list(np.random.randint(10, 14, cor_n_features)) + rand_data[d][cor_n_features+ cor_n_features:]
        else:
            rand_data[d] = rand_data[d][:cor_n_features] + list(np.random.randint(10, 14, 5)) + rand_data[d][cor_n_features+ 5:]

    if sum(y[drift_loc:]) >= class_ratio*(len(rand_data)-drift_loc):
        return rand_data, y

    new_cnt = int(class_ratio*(len(rand_data)-drift_loc)) - sum(y[drift_loc:])
    idx =  random.randint(drift_loc, len(rand_data)-1)
    while new_cnt != 0:
        d = rand_data[idx]
        if all(val == new_feat for val in d[:cor_n_features]):
                idx = random.randint(drift_loc,len(rand_data)-1)
                continue 
        else:
            rand_data[idx] = [new_feat]*cor_n_features + rand_data[idx][cor_n_features:]
            y[idx] = 1
            new_cnt-=1
            idx = random.randint(drift_loc,len(rand_data)-1)

    return rand_data, y

def two_interact_y_new(rand_data, cor_n_features, drift_loc):
    class_ratio = 0.1
    y = []
    for d in rand_data:
        if all(val == 0 for val in d[:cor_n_features]):
            y.append(1)
        elif all(val == 1 for val in d[1:cor_n_features+1]):
            y.append(1)
        else:
            y.append(0)

    if sum(y[:drift_loc]) >= class_ratio*drift_loc:
        return rand_data, y

    new_cnt = int(class_ratio*drift_loc) - sum(y[:drift_loc])
    idx = random.randint(0,drift_loc-1)
    while new_cnt != 0:
        d = rand_data[idx]
        if all(val == 0 for val in d[:cor_n_features]):
                idx = random.randint(0,drift_loc-1)
                continue 
        elif all(val == 1 for val in d[1:cor_n_features+1]):
                idx = random.randint(0,drift_loc-1)
                continue 
        else:
            r = random.randint(0,1)
            if r == 0:
                rand_data[idx] = [0]*cor_n_features + rand_data[idx][cor_n_features:]
            else:
                rand_data[idx] = rand_data[idx][:1] + [1]*cor_n_features + rand_data[idx][cor_n_features+1:]
            y[idx] = 1
            new_cnt-=1
            idx = random.randint(0,drift_loc-1)

    return rand_data, y

def two_interact_y_drifted_new(rand_data, cor_n_features, drift_loc):
    class_ratio = 0.1
    y = []
    for d in rand_data:
        if all(val == 1 for val in d[:cor_n_features]):
            y.append(1)
        elif all(val == 0 for val in d[1:cor_n_features+1]):
            y.append(1)
        else:
            y.append(0)

    for d in range(len(rand_data)):
        if 2 * cor_n_features + 1 > len(rand_data[0]):
            useless_f_cnt = cor_n_features - ((2 * cor_n_features + 1) - len(rand_data[0]))
            if useless_f_cnt < 0:
                useless_f_cnt = 0
        else:
            useless_f_cnt = cor_n_features
        rand_data[d] = rand_data[d][:cor_n_features+1] + list(np.random.randint(10, 14, useless_f_cnt)) + rand_data[d][cor_n_features+1+ useless_f_cnt:]

    if sum(y[drift_loc:]) >= class_ratio*(len(rand_data)-drift_loc):
        return rand_data, y

    new_cnt = int(class_ratio*(len(rand_data)-drift_loc)) - sum(y[drift_loc:])
    idx = random.randint(drift_loc,len(rand_data)-1)
    while new_cnt != 0:
        d = rand_data[idx]
        if all(val == 2 for val in d[:cor_n_features]):
                idx = random.randint(drift_loc,len(rand_data)-1)
                continue 
        elif all(val == 3 for val in d[1:cor_n_features+1]):
                idx = random.randint(drift_loc,len(rand_data)-1)
                continue 
        else:
            r = random.randint(0,1)
            if r == 0:
                rand_data[idx] = [2]*cor_n_features + rand_data[idx][cor_n_features:]
            else:
                rand_data[idx] = rand_data[idx][:1] + [3]*cor_n_features + rand_data[idx][cor_n_features+1:]
            y[idx] = 1
            new_cnt-=1
            idx = random.randint(drift_loc,len(rand_data)-1)
    return rand_data, y

def gradual_shift_new(x, y, drift_loc, w, concepts):
    drifted_x = []
    drifted_y = []
    if concepts == 1:
        x_new, y_new = single_interact_y_drifted_new(x.copy(), cor_n_features, drift_loc)
    else:
        x_new, y_new = two_interact_y_drifted_new(x.copy(), cor_n_features, drift_loc)
    
    for i in range(len(x)):
        prob = shift_prob(i, drift_loc, w)
        rand_num = np.random.rand()
        #print(rand_num, prob)
        if rand_num > prob:
            drifted_x.append(x[i])
            drifted_y.append(y[i])
        else:
            drifted_x.append(x_new[i])
            drifted_y.append(y_new[i])

    return drifted_x, drifted_y

def shift_prob(t, t_d, w):
    p_t = 1 / (1 + np.exp(-4*(t-t_d)/w))
    return p_t

def gradual_shift(x, y, drift_loc, w, concepts):
    drifted_x = []
    drifted_y = []

    for i in range(len(x)):
        prob = shift_prob(i, drift_loc, w)
        rand_num = np.random.rand()
        if rand_num > prob:
            drifted_x.append(x[i])
            drifted_y.append(y[i])
        else:
            if concepts == 1:
                drifted_inst, drift_label = single_interact_y_drifted(x[i], cor_n_features)
            else:
                drifted_inst, drift_label = two_interact_y_drifted(x[i], cor_n_features)
            drifted_x.append(drifted_inst)
            drifted_y.append(drift_label)

    return drifted_x, drifted_y

def get_ale_f_imp(drift, x, y, feature_name, feat_idx):
    predictor = RandomForestRegressor(random_state=seed)
    predictor.fit(x, y)
    prediction_fn = lambda x: predictor.predict(x)
    print('Train score: %.2f' % (predictor.score(x, y)))

    explainer = ALE(predictor=prediction_fn, feature_names=feature_name)
    pd_std = []
    feat_idx = feat_idx[:len(feature_name)]
    for i in feat_idx:
        print("Explaining:", i)
        exp = explainer.explain(X=x,
                                features=[i])
        for i in exp.ale_values:
            pd_std.append(np.std(exp.ale_values))

    single_feat_imp = np.array(pd_std[:toltal_n_features])*-1

    single_feat_imp_ranking = np.argsort(single_feat_imp)

    k = 10
    top_k_single_feat = single_feat_imp_ranking[:k]


    fig = plt.figure(figsize = (10, 5))
    if drift == "_drift":
        c = 'maroon'
    else:
        c = 'green'

    plt.bar([str(i) for i in top_k_single_feat], [single_feat_imp[i]*-1 for i in top_k_single_feat], color = c,
            width = 0.4, label = drift[1:])

    plt.xlabel("Features")
    plt.ylabel("Feature Imp")
    plt.title("Top 10 ALE Features")
    plt.legend()
    plt.show()


    return [single_feat_imp[i]*-1 for i in top_k_single_feat], top_k_single_feat

def get_pdp_f_imp(drift, x, y, feature_name, feat_idx, 
                  trained_predictor, k = 10):

    single_pdp_start = time.time()
    prediction_fn = lambda x: trained_predictor.predict(x)
    global time_exceeded
    if time_exceeded == True:
        time_exceeded = False
        return [], []
    explainer = PartialDependence(predictor=prediction_fn,
                        feature_names=feature_name)

    pd_std = []
    print("Getting PDP Features Importance...")
    for i in tqdm(feat_idx[:len(feature_name)]):
        exp = explainer.explain(X=x,
                                features=[i],
                                kind='average',
                                grid_resolution = 25)
        for i in exp.pd_values:
            pd_std.append(np.std(exp.pd_values))

    cor_size = len(feat_idx[-1])

    single_feat_imp = np.array(pd_std[:toltal_n_features])*-1
    single_pdp_end = time.time()

    print("Getting PDP Feature Comb Importance...")
    for i in tqdm(feat_idx[len(feature_name):]):
        pdp_t_spent = time.time() - single_pdp_start
        if pdp_t_spent > timer:
            print("Time Exceeded")
            global pdp_pass
            pdp_pass = True
            time_exceeded = True
            return [], []
        exp = explainer.explain(X=x,
                                features=[i],
                                kind='average',
                                grid_resolution = 25)
        for i in exp.pd_values:
            pd_std.append(np.std(exp.pd_values))

    cor_size = len(feat_idx[-1])

    feat_comb_imp = np.array(pd_std[toltal_n_features:])*-1

    return single_feat_imp*-1, feat_comb_imp*-1

def get_shap_f_imp_new(drift, x, y, feature_name, feat_combs, 
                  trained_predictor, k = 10, repeats = 10):
    if drift == "_drift":
        c = 'maroon'
    else:
        c = 'green'

    single_shap_start = time.time()

    global time_exceeded
    if time_exceeded == True:
        time_exceeded = False
        return [], []
    x_df = pd.DataFrame(x, columns = feature_name)

    n_features = len(feature_name)
    single_feat_imp = []
    print("Getting SHAP features Importance...")
    for feature_set in tqdm(feat_idx[:n_features]):

        marginal_contributions_all = []
        sample_list = random.choices(range(len(x)), k=repeats)
        for sample_idx in sample_list:
            #sample_idx = 0
            sample_x = x_df.iloc[sample_idx]
            marginal_contributions = []
            feature_idxs = list(range(n_features))

            for feature in [feature_set]:
                feature_idxs.remove(feature)

            for _ in range(repeats):
                z = x_df.sample(1).values[0]
                x_idx = random.sample(feature_idxs, random.choice(range(len(feature_idxs))))
                z_idx = [idx for idx in feature_idxs if idx not in x_idx]

                # construct two new instances
                x_plus_j = np.array([sample_x[i] if i in x_idx + [feature_set] else z[i] for i in range(n_features)])
                x_minus_j = np.array([z[i] if i in z_idx + [feature_set] else sample_x[i] for i in range(n_features)])
                
                # calculate marginal contribution
                marginal_contribution = trained_predictor.predict_proba(x_plus_j.reshape(1, -1))[0][1] - trained_predictor.predict_proba(x_minus_j.reshape(1, -1))[0][1]
                marginal_contributions.append(marginal_contribution)
            phi_j_x = sum(marginal_contributions) / len(marginal_contributions)  # our shaply value

            marginal_contributions_all.append(phi_j_x)
        single_feat_imp.append(np.mean(marginal_contributions_all))
        #print(f"Shaply value for feature j: {phi_j_x:.5}")

    #print(single_feat_imp)
    single_feat_imp = np.array(single_feat_imp)*-1
    single_shap_end = time.time()

    print("Getting SHAP Features Comb Importance...")
    feat_comb_imp = []
    for feature_set in tqdm(feat_combs):
        shap_t_spent = time.time() - single_shap_start
        if shap_t_spent > timer:
            print("Time Exceeded")
            global shap_pass
            shap_pass = True
            time_exceeded = True
            return [], []
        marginal_contributions_all = []
        sample_list = random.choices(range(len(x)), k=repeats)
        for sample_idx in sample_list:
            sample_x = x_df.iloc[sample_idx]
            marginal_contributions = []
            feature_idxs = list(range(n_features))

            for feature in feature_set:
                feature_idxs.remove(feature)

            for _ in range(repeats):
                z = x_df.sample(1).values[0]
                #x_idx = random.sample(feature_idxs, min(max(int(0.2*n_features), random.choice(feature_idxs)), int(0.8*n_features)))
                x_idx = random.sample(feature_idxs, random.choice(range(len(feature_idxs))))
                z_idx = [idx for idx in feature_idxs if idx not in x_idx]

                # construct two new instances
                x_plus_j = np.array([sample_x[i] if i in x_idx + list(feature_set) else z[i] for i in range(n_features)])
                x_minus_j = np.array([z[i] if i in z_idx + list(feature_set) else sample_x[i] for i in range(n_features)])
                
                # calculate marginal contribution
                marginal_contribution = trained_predictor.predict_proba(x_plus_j.reshape(1, -1))[0][1] - trained_predictor.predict_proba(x_minus_j.reshape(1, -1))[0][1]
                marginal_contributions.append(marginal_contribution)
            phi_j_x = sum(marginal_contributions) / len(marginal_contributions)  # our shaply value

            marginal_contributions_all.append(phi_j_x)
        feat_comb_imp.append(np.mean(marginal_contributions_all))
    
    feat_comb_imp = np.array(feat_comb_imp)*-1

    return single_feat_imp*-1, feat_comb_imp * -1

def get_permute_feature_importance(drift, x, y, feature_name, feat_idx,
                                   trained_predictor, k = 10, repeats = 10):
    if drift == "_drift":
        c = 'maroon'
    else:
        c = 'green'

    single_pfi_start = time.time()

    global time_exceeded
    if time_exceeded == True:
        time_exceeded = False
        return [], []
    ref_score = trained_predictor.score(x, y)
    n_features = x.shape[1]
    perm_imp = []
    
    print("Getting PFI Features Importance...")
    for feature_set in tqdm(feat_idx[:n_features]):
        X_permuted = x.copy()
        f_score_lst = []
        for rep in range(repeats):
            for feature in [feature_set]:
                X_permuted[:, feature] = np.random.permutation(X_permuted[:, feature])
            f_score_lst.append(trained_predictor.score(X_permuted, y))
        avg_score = np.mean(f_score_lst)
        perm_imp.append(np.abs(ref_score - avg_score))

    cor_size = len(feat_idx[-1])

    single_feat_imp = np.array(perm_imp[:n_features])*-1

    single_pfi_end = time.time()

    print("Getting PFI Feature Combs Importance...")
    for feature_set in tqdm(feat_combs):
        shap_t_spent = time.time() - single_pfi_start
        if shap_t_spent > timer:
            print("Time Exceeded")
            global pfi_pass
            pfi_pass = True
            time_exceeded = True
            return [], []
        X_permuted = x.copy()
        f_score_lst = []
        for rep in range(repeats):
            for feature in feature_set:
                X_permuted[:, feature] = np.random.permutation(X_permuted[:, feature])
            f_score_lst.append(trained_predictor.score(X_permuted, y))
        avg_score = np.mean(f_score_lst)
        perm_imp.append(np.abs(ref_score - avg_score))

    feat_comb_imp = np.array(perm_imp[n_features:])*-1
    return single_feat_imp*-1, feat_comb_imp*-1

def plt_dif(f_org, f_drift, f_comb_org, f_comb_drift, 
            f_name, name_single, name_comb):
    
    if len(f_comb_org) == 0 or len(f_comb_drift)  == 0:
        return [], []
    
    dif_single = f_org - f_drift
    df_single = pd.DataFrame({'org': f_org, 'drift': f_drift, 
                            'shift': dif_single}, index = name_single)
    sns.set_theme(style="whitegrid")
    plt.rcParams.update({'font.size': 22})
    color_list = ['grey', 'black', 'red']

    df_single.sort_values(by=['shift'], ascending=False, key = lambda col: abs(col)).head(10).plot.bar(rot=0, 
                    width = 0.8, color = color_list, figsize=(6,3))
    plt.xlabel("Features")
    plt.ylabel("Importance")
    sns.despine(left=True)
    plt.title(f_name)
    plt.legend(loc='center right')
    plt.show()

    dif_comb = f_comb_org - f_comb_drift
    df_comb= pd.DataFrame({'org': f_comb_org, 'drift': f_comb_drift, 
                        'shift': dif_comb}, index = name_comb)
    df_comb.sort_values(by=['shift'], ascending=False, key = lambda col: abs(col)).head(10).plot.bar(rot=45, 
                    width = 0.8, color = color_list, figsize=(6,3))
    plt.xlabel("Features Combinations")
    plt.ylabel("Importance")
    sns.despine(left=True)
    plt.title(f_name)
    plt.legend(loc='center right')
    #plt.xticks(fontsize=8)
    plt.show()

    return df_single, df_comb

seed = 0
n_samples = 50000
drift_loc =int(n_samples/2)
w = 50
data_generator = "Rand"
#data_generator = "Agrawal"

random.seed(seed)
np.random.seed(seed)

timer = 86400
time_exceeded = False
pdp_pass = False
shap_pass = False
pfi_pass = False

k_list = [2,3,4,6,8,10]
# k_list = [2]
for c_f in k_list:
    #* Random Data Generation
    if data_generator == "Rand":
        cor_n_features = c_f
        toltal_n_features = 20
        concept_cnt = 2
        blanced = True

        x, feature_name = generate_data(n_samples, toltal_n_features, 3, seed)
        
        if blanced:
                    #* Generate class labels where the label is 1 the first cor n feature are 0
            if concept_cnt == 1:
                data_name = "rand_single_drift_" + str(cor_n_features)
                x, y = single_interact_y_new(x, cor_n_features, drift_loc)
            else:
                    #* Generate class labels where the label is 1 the first cor n feature are 0
                    #* and the cor n+1 feature are 1
                data_name = "rand_two_drift_" + str(cor_n_features)
                x, y = two_interact_y_new(x, cor_n_features, drift_loc)

            drifted_x, drifted_y = gradual_shift_new(x, y, drift_loc, w, concept_cnt)

        else:
                            #* Generate class labels where the label is 1 the first cor n feature are 0
            if concept_cnt == 1:
                data_name = "rand_single_drift_" + str(cor_n_features)
                y = single_interact_y(x, cor_n_features)
            else:
                    #* Generate class labels where the label is 1 the first cor n feature are 0
                    #* and the cor n+21feature are 1
                data_name = "rand_two_drift_" + str(cor_n_features)
                y = two_interact_y(x, cor_n_features)

            drifted_x, drifted_y = gradual_shift(x, y, drift_loc, w, concept_cnt)


            #! cor_n_features = 2 by default
        feat_idx = list(range(toltal_n_features))
        feat_combs = []
        for comb in itertools.combinations(feat_idx, cor_n_features):
            feat_combs.append(comb)
        feat_idx.extend(feat_combs)

        name_single = [str(i) for i in range(toltal_n_features)]
        name_comb = []
        for i in feat_combs:
            str_comb = '{'
            for j in range(len(i)):
                if j != cor_n_features-1:
                    str_comb += str(i[j])
                    str_comb += ','
                else:
                    str_comb += str(i[j])
            str_comb += '}'
            name_comb.append(str_comb)

    #* AGRAWAL Data Generation
    if data_generator == "Agrawal":
        s_name = c_f
        if s_name == 1:
            fnc_org = 2
            fnc_drifted = 3
            cor_n_features = 2
        elif s_name == 2:
            fnc_org = 1
            fnc_drifted = 11
            cor_n_features = 2
        elif s_name == 3:
            fnc_org = 2
            fnc_drifted = 12
            cor_n_features = 2
        elif s_name == 4:
            fnc_org = 3
            fnc_drifted = 13
            cor_n_features = 3
        data_name = "AGR_" + str(fnc_org) + "_" + str(fnc_drifted)
        drifted_x, drifted_y, feature_name = generate_argawal(n_samples, drift_loc, w, seed, fnc_org, fnc_drifted)
        toltal_n_features = len(feature_name)

            #! cor_n_features = 2 by default
        feat_idx = list(range(toltal_n_features))
        feat_combs = []
        for comb in itertools.combinations(feat_idx, cor_n_features):
            feat_combs.append(comb)
        feat_idx.extend(feat_combs)

        name_single = feature_name
        name_comb = []
        for i in feat_combs:
            str_comb = '{'
            for j in range(len(i)):
                if j != cor_n_features-1:
                    str_comb += str(i[j])
                    str_comb += ','
                else:
                    str_comb += str(i[j])
            str_comb += '}'
            name_comb.append(str_comb)

    detected_drift_loc = drift_loc

    ref_window_size = drift_loc
    org_x = np.array(drifted_x[:ref_window_size])
    org_y = np.array(drifted_y[:ref_window_size])
    new_x = np.array(drifted_x[detected_drift_loc: detected_drift_loc + ref_window_size])
    new_y = np.array(drifted_y[detected_drift_loc: detected_drift_loc + ref_window_size])

    print(data_name)
    print("Class Distribution:", Counter(org_y), Counter(new_y))
    predictor_org = RandomForestClassifier(random_state=seed)
    predictor_new = RandomForestClassifier(random_state=seed)
    predictor_org.fit(org_x, org_y)
    predictor_new.fit(new_x, new_y)
    #* Get PDP Feature Importance 
    f_name = 'PDP'
    def get_pdp():
        pdp_f_imp_org, pdp_f_comb_imp_org = get_pdp_f_imp("_org", org_x, org_y, feature_name,feat_idx, predictor_org)
        pdp_f_imp_drift, pdp_f_comb_imp_drift = get_pdp_f_imp("_drift",new_x, new_y, feature_name, feat_idx, predictor_new)
        pdp_single_df, pdp_comb_df = plt_dif(pdp_f_imp_org, pdp_f_imp_drift, pdp_f_comb_imp_org, pdp_f_comb_imp_drift, 
                f_name, name_single, name_comb)
    if pdp_pass is False:
         get_pdp()


    #* Get SHAP Feature Importance 
    f_name = 'SHAP'
    def get_shap():
        shap_f_imp_org, shap_f_interact_org = get_shap_f_imp_new("_org", org_x, org_y, feature_name, feat_combs, predictor_org)
        shap_f_imp_drift, shap_f_interact_drift= get_shap_f_imp_new("_drift", new_x, new_y, feature_name, feat_combs, predictor_new)
        shap_single_df, shap_comb_df = plt_dif(shap_f_imp_org, shap_f_imp_drift, shap_f_interact_org, shap_f_interact_drift, 
                f_name, name_single, name_comb)
    
    if shap_pass is False:
         get_shap()
    #* Get PFI Feature Importance 
    f_name = 'PFI'
    def get_pfi():
        perm_f_imp_org, perm_f_comb_org = get_permute_feature_importance("_org", org_x, org_y, feature_name, feat_idx, predictor_org)
        perm_f_imp_drift, perm_f_comb_drift = get_permute_feature_importance("_drift",new_x, new_y, feature_name, feat_idx, predictor_new)
        perm_single_df, perm_comb_df = plt_dif(perm_f_imp_org, perm_f_imp_drift, perm_f_comb_org, perm_f_comb_drift, 
                f_name, name_single, name_comb)
    if pfi_pass is False:
         get_pfi()
