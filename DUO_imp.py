import numpy as np
import pandas as pd
import random
import numpy as np
from collections import Counter
from skmultiflow.data import AGRAWALGenerator
from DUO_map import duo_model, analyze_shared_feature
from efficient_apriori import apriori
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations, chain
from scipy.special import comb
from tqdm import tqdm
import time

from tensorflow import keras
from keras import layers
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from keras import backend as K

import warnings
warnings.simplefilter("ignore", UserWarning)
import tensorflow as tf
from tensorflow.python.framework.ops import disable_eager_execution
disable_eager_execution()
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

def plt_loss(model, e):
    epochs = range(e)
    loss = model.history['loss']
    val_loss = model.history['val_loss']
    plt.plot(epochs, loss, label='Training loss')
    plt.plot(epochs, val_loss, label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.show()

def comb_index(n, k):
    count = comb(n, k, exact=True)
    index = np.fromiter(chain.from_iterable(combinations(range(n), k)), 
                        int, count=count*k)
    return index.reshape(-1, k)

def get_key_color(key_name):
    cmap = matplotlib.cm.get_cmap('Accent')
    if key_name == "abnorm_norm_imp" or key_name == 0:
        c = cmap(0)
    elif key_name == "abnorm_norm_unimp" or key_name == 1:
        c = cmap(0.33)
    elif key_name == "abnorm_imp_unimp" or key_name == 2:
        c = cmap(0.67)
    else:
        c = cmap(0.99)
    return c

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
        f_names.append(str(i))

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
    gen =  AGRAWALGenerator(random_state = 42)
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

def shift_prob(t, t_d, w):
    p_t = 1 / (1 + np.exp(-4*(t-t_d)/w))
    return p_t

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

def half_ae(org_x, org_y, new_x, new_y, b, e, task_type = 'classification'):
    scaler_x = MinMaxScaler()
    x = np.concatenate((org_x, new_x), axis=0)
    x = scaler_x.fit_transform(x)
    y = np.concatenate((org_y, new_y), axis=0)
    org_x = x[:len(org_x)]
    new_x = x[len(org_x):]

    if task_type != 'classification':
        y = np.sqrt(y)
        org_y = np.sqrt(org_y)
        new_y = np.sqrt(new_y)


     #* Create AE Model
    input = keras.Input(shape=(org_x.shape[1],))
    encoded1 = layers.Dense(16, activation='elu')(input)
    encoded2 = layers.Dense(8, activation='elu')(encoded1)

    decoded = layers.Dense(16, activation='elu')(encoded2)
    decoded = layers.Dense(org_x.shape[1], activation='elu')(decoded)

    autoencoder = keras.Model(input, decoded)
    autoencoder.compile(optimizer='adam', loss='mse')
    #autoencoder.summary() 
    autoencoder_train = autoencoder.fit(org_x, org_x,
                    epochs=e,
                    batch_size=b,
                    verbose = 0,
                    validation_split = 0.2)
    plt_loss(autoencoder_train, e)


    #* Get Org Reconstruction Error loss.
    x_train_pred = autoencoder.predict(x)
    train_mae_loss = np.mean(np.square(x_train_pred - x), axis=1)
    plt.figure(figsize=(6, 4))
    sns.set_theme(style="whitegrid")
    plt.hist(train_mae_loss, bins=100)
    plt.xlabel("Org Reconstruction(MSE) loss")
    plt.ylabel("No of samples")
    plt.title("Epochs:" +  str(e))
    sns.despine(left=True)
    plt.show()

    cf1 = layers.Dense(8, activation='elu')(encoded2)
    if task_type == 'classification':
        cf2 = layers.Dense(1, activation='sigmoid')(cf1)
        l = 'binary_crossentropy'
    else:
        cf2 = layers.Dense(1, activation='elu')(cf1)
        l = 'mse'
    model = keras.Model(input, cf2)

    for l1,l2 in zip(model.layers[:3],autoencoder.layers[0:3]):
        l1.set_weights(l2.get_weights())

    for layer in model.layers[0:3]:
        layer.trainable = False
    model.compile(optimizer='adam', loss=l)
    #model.summary() 

    classifier = model.fit(org_x, org_y, epochs=e, validation_split = 0.2, verbose = 0,batch_size=b)
    plt_loss(classifier,e)

    # #* Get Classification loss
    # y_train_pred = model.predict(x)

    #* Fine Tuning the Classifier
    for layer in model.layers[0:3]:
        layer.trainable = True
    model.compile(optimizer=keras.optimizers.legacy.Adam(1e-4), loss=l)

    tuned_classifier = model.fit(org_x, org_y, epochs=e, validation_split = 0.2, verbose = 0, batch_size=b)
    plt_loss(tuned_classifier,e)

    y_train_pred_tuned = model.predict(x).flatten()

    imp = np.absolute(y-y_train_pred_tuned)
    plt.figure(figsize=(6, 4))
    sns.set_theme(style="whitegrid")
    plt.hist(imp, bins=100)
    plt.xlabel("Prediction difference")
    plt.ylabel("No of samples")
    plt.title("Epochs:" +  str(100))
    sns.despine(left=True)
    plt.show()

    return train_mae_loss, imp, x_train_pred

def get_norm_imp(train_mae_loss, imp, y, norm_offset, imp_offset):

    norm_imp_df = pd.DataFrame({'Reco_Loss': train_mae_loss, 'Pred_Loss': imp})
    features = list(norm_imp_df.columns.values)
    for f in features:
        if f == 'Reco_Loss':
            norm_imp_df[f] = pd.cut(norm_imp_df[f], 100, labels=False)
            norm_imp_df[f+'_norm'] = pd.cut(norm_imp_df[f], 100, labels=False)
        else:
            norm_imp_df[f] = pd.cut(norm_imp_df[f], 100, labels=False)
            norm_imp_df[f+'_norm'] = pd.cut(norm_imp_df[f], 100, labels=False)

    most_common_bins = norm_imp_df.mode()
    m_norm = []
    for i in norm_imp_df['Reco_Loss'].values:
        threshold = most_common_bins['Reco_Loss'].values[0]
        
        if i > threshold + norm_offset:
            m_norm.append(-1)
        else:
            m_norm.append(1)
    #print("Reco_Loss Treshold:", threshold)

    m_imp = []
    for i in norm_imp_df['Pred_Loss'].values:
        threshold = most_common_bins['Pred_Loss'].values[0]
        if i > threshold + imp_offset:
            m_imp.append(-1)
        else:
            m_imp.append(1)
    #print("Pred_Loss Treshold:", threshold)
    return m_norm, m_imp

def get_norm_imp_topk(train_mae_loss, imp, k_cnt = 2000):

    if k_cnt > len(train_mae_loss):
        k_cnt = len(train_mae_loss)

    #! -1: Abnormal, Important; +1: Normal, Unimportant
    m_norm = []
    sroted_reco_Loss_idx = np.argsort(train_mae_loss)
    top_k_reco_Loss_idx  = sroted_reco_Loss_idx[-k_cnt:]
    m_norm = [1]*len(train_mae_loss)
    for i in top_k_reco_Loss_idx:
        m_norm[i] = -1


    m_imp = []
    sroted_imp_idx = np.argsort(imp)
    top_k_imp_idx  = sroted_imp_idx[-k_cnt:]
    m_imp = [1]*len(imp)
    for i in top_k_imp_idx:
        m_imp[i] = -1

    return m_norm, m_imp

def get_rules(f_comb_df, feature_name, sup, conf, rule_len, verbose= False):
    group_names = f_comb_df.columns.values.tolist()
    group_transac = []
    rules_list = [[],[],[],[]]
    conf_list = [[],[],[],[]]
    f_heat_map_list = np.array([np.zeros(len(feature_name))]*4)

    if len(feature_name) < 100:
        number_of_rules = None
        max_rule_len = cor_n_features
    else:
        number_of_rules = None
        group_names = group_names[2:-1]
        max_rule_len = 5

    for q,g in enumerate(group_names):
        if q == 0:
            continue
        tempdf = f_comb_df[[g]]
        tempdf = tempdf.loc[tempdf[g] != 0]
        transac = []
        occurnce_cont = tempdf[g].values.tolist()
        for i,comb in enumerate(tempdf.index.values):
            comb = comb.split(',')
            if verbose is True:
                comb = [feature_name[int(n)] for n in comb]
            c = occurnce_cont[i]
            for j in range(int(c)):
                transac.append(comb)
        transac = [tuple(t) for t in transac]
        print(g, len(transac))
        group_transac.append(transac)
        itemsets, rules = apriori(transac, min_support=sup, min_confidence=conf, max_length = cor_n_features, verbosity= 1)

        if rule_len != 'all':
            rules = filter(lambda rule: len(rule.lhs) + len(rule.rhs) <= rule_len, rules)

        print("Filtering Rules..")
        for rule in tqdm(sorted(rules, key=lambda rule: (rule.confidence + rule.support)*-1)[:number_of_rules]):
            #print(rule) 
            if number_of_rules != None:
                print(rule) 
            f_in_rule = set(rule_to_list(rule, verbose, feature_name))

            if f_in_rule not in rules_list[q]:
                metric = rule.confidence + rule.support
                rules_list[q].append(f_in_rule)
                conf_list[q].append(metric)
                for f in f_in_rule:
                    f_heat_map_list[q][f] += metric
        print("")
        print("")
        rules_list, conf_list = get_k_comb(rules_list, conf_list, rule_len)
    return rules_list, conf_list, f_heat_map_list

def get_k_comb(rules_list, conf_list, k):
    if k == 'All':
        return k_rules_list, k_conf_list
    else:
        k_rules_list = [[],[],[],[]]
        k_conf_list = [[],[],[],[]]
        for g_id, g in enumerate(rules_list):
            for rule_id, rule in enumerate(g):
                if len(rule) == k:
                    k_rules_list[g_id].append(rule)
                    k_conf_list[g_id].append(conf_list[g_id][rule_id])
        return k_rules_list, k_conf_list

def rule_to_list(rule, v, feature_name):
    rule = str(rule).split(' ')
    rule = rule[:-8]
    rule.remove('->')
    rule[0] = rule[0][1:]
    if rule[-1][0] == "{":
        rule[-1] = rule[-1][1:]

    #print(rule)
    if v is False:
        for i in range(len(rule)):
            if rule[i][0] == "{":
                rule[i] = int(rule[i][1:-1])
            else:
                rule[i] = int(rule[i][:-1])
    else:
        for i in range(len(rule)):
            rule[i] = feature_name.index(rule[i][:-1])
    return rule

seed = 0
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

n_samples = 50000
drift_loc =int(n_samples/2)
w = 50
ref_window_size = drift_loc


data_generator = "Rand"
#data_generator = "Agrawal"


#* Random Data Generation
if data_generator == "Rand":
    cor_n_features = 3
    toltal_n_features = 10
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

#* AGRAWAL Data Generation
if data_generator == "Agrawal":
    s_name = 4
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
    toltal_n_features = 9
    data_name = "AGR_" + str(fnc_org) + "_" + str(fnc_drifted)
    drifted_x, drifted_y, feature_name = generate_argawal(n_samples, drift_loc, w, seed, fnc_org, fnc_drifted)

detected_drift_loc = drift_loc
org_x = np.array(drifted_x[:ref_window_size])
org_y = np.array(drifted_y[:ref_window_size])
new_x = np.array(drifted_x[detected_drift_loc: detected_drift_loc + ref_window_size])
new_y = np.array(drifted_y[detected_drift_loc: detected_drift_loc + ref_window_size])

print('Positive Count:', 'Org', Counter(org_y), 'Drifted', Counter(new_y))


 #*Train AE
duo_start = time.time()
epoch = 100
batch = int(0.01*len(drifted_y))
ae_mode = 'half'
# mode = "nearest"
number_of_days = len(drifted_y)
train_mae_loss, y_train_pred_tuned, x_train_pred = half_ae(org_x, org_y, new_x, new_y, batch, epoch)

 #*Get Feature Imp
def get_cor_rules(x, y, norm_offset, imp_offset, sup, conf, rule_size, norm_dist = False):
    #* Get DUO model grouping (top-k or threshold)
    # m_norm, m_imp = get_norm_imp_topk(train_mae_loss, y_train_pred_tuned, k_cnt = 2000)
    m_norm, m_imp = get_norm_imp(train_mae_loss, y_train_pred_tuned, y, norm_offset, imp_offset)
    if norm_dist:
        m_norm = np.concatenate((np.array([1]*ref_window_size),np.array([-1]*ref_window_size)), axis=0)

    print("Normal:", Counter(m_norm), "Imp:", Counter(m_imp))
    print("Ref Data Norm:", Counter(m_norm[:ref_window_size]), "Imp:", Counter(m_imp[:ref_window_size]))
    print("Drifted Data Norm:", Counter(m_norm[ref_window_size:]), "Imp:", Counter(m_imp[ref_window_size:]))
    #* Analyse Features
    data_binned, result, cmmn_bins, feature_map_keys, key_cnts = duo_model(m_norm, m_imp, x, 10, feature_name)
    print("Analyzing feature map")
    print("Extracting shared feature map")
    f_comb_df,f_comb_idx_df  = analyze_shared_feature(len(feature_name), data_binned, cmmn_bins, feature_map_keys)
    

    #* Get frequent feature sets from shared feature space
    print("Getting Rules..")
    rules_list, conf_list, f_heat_map_list = get_rules(f_comb_df, feature_name, sup, conf, rule_size, verbose = False)
    group_names = f_comb_df.columns.values.tolist()

    f_single_df = pd.DataFrame(f_heat_map_list.T, columns = group_names, index = feature_name)
    small_f_single_df = f_single_df.iloc[:,[1,2]]
    small_f_single_df.columns = ['Non-contributing', 'Contributing']
    fig, axes = small_f_single_df.sort_values(by=['Contributing', 'Non-contributing'], ascending=False).head(10).plot.bar(rot=0, 
                color = [get_key_color(1), get_key_color(2)],width = 0.8, 
                figsize=(6,3), subplots=True, title=['', ''], legend = False)
    plt.xlabel("Features")
    fig.text(-1.7, 0.5, 'Importance', va='center', rotation='vertical')
    sns.despine(left=True)
    plt.suptitle("DUO")
    plt.figlegend(loc='center right', bbox_to_anchor=(0.9, 0.4))
    plt.show()

    top_k_f_conf_small = []
    top_k_f_comb_str_small = []
    for i,rule in enumerate(rules_list):
        top_k_f_comb = rule[:10]
        top_k_f_conf = conf_list[i][:10]
        top_k_f_conf = [val/2 for val in top_k_f_conf]
        top_k_f_conf_small.append(top_k_f_conf)

        top_k_f_comb_str = []
        for f_comb in top_k_f_comb:
            f_comb = list(f_comb)
            comb_str = ''
            for f in range(len(f_comb) - 1):
                comb_str += str(f_comb[f])
                comb_str += ','
            comb_str += str(f_comb[-1])
            top_k_f_comb_str.append(comb_str)
        top_k_f_comb_str_small.append(top_k_f_comb_str)

    top_k_f_conf_small = top_k_f_conf_small[1:3]
    top_k_f_comb_str_small = top_k_f_comb_str_small[1:3]
    fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(6,3))
    fig.subplots_adjust(hspace = .9)
    p = 1
    j = 0
    group = ['Non-contributing', 'Contributing']
    for g, x_plt, y_plt, ax in zip(group, top_k_f_comb_str_small, top_k_f_conf_small, axs.ravel()):
        x_plt = ['{' + i + '}' for i in x_plt]
        ax.bar(x_plt, y_plt, label = g,
                width = 0.4, color = get_key_color(p))
        ax.tick_params(axis='x', labelrotation=45)
        p+=1
    plt.setp(axs, ylim = (0, 1))
    sns.despine(left=True)
    plt.suptitle("DUO")
    lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes]
    lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
    fig.legend(lines, labels, loc='center right', bbox_to_anchor=(0.9, 0.33))
    plt.xlabel("Feature Combinations")
    fig.text(0.03, 0.5, 'Importance', va='center', rotation='vertical')
    plt.show()

    return rules_list, conf_list, f_heat_map_list, f_comb_df

x = np.concatenate((org_x, new_x), axis=0)
y = np.concatenate((org_y, new_y), axis=0)


rules_list, conf_list, f_heat_map_list, f_comb_df = get_cor_rules(x, y, norm_offset = 20, imp_offset = 0, 
                                                                  sup = 0.6, conf = 0.8, 
                                                                  rule_size = cor_n_features, norm_dist = False)

