import numpy as np
from matplotlib.colors import cnames
from pandas import DataFrame
from matplotlib import pyplot as plt
from sklearn import metrics
# Author list: 
    # 1. https://orcid.org/0000-0001-9996-6638
# Chronology:
    # 01.01.2022 - v2


# BEGIN
# Generates a simple random dataset to classification
# 
def gen_dataset(N_points=100, N_groups = 2, N_features=2):
    
    features = ['feature %s'%(n+1) for n in range(N_features)]
    group_names = ['group %s'%(n+1) for n in range(N_groups)]
    
    dataset = DataFrame(columns= features+['tag','name'])

    bias_points = {group: np.random.uniform(-2*np.sqrt(N_groups), 
                                            2*np.sqrt(N_groups), 
                                            N_features) for group in group_names}
    
    for n in range(N_points):
        for n, group_name in enumerate(group_names):
            numbers = np.random.normal(0.0, 1.0, N_features)+bias_points[group_name]
            dataset.loc[len(dataset)+1] = list(numbers) + [n, group_name]
    
    return dataset


# It will colorize an array you want to colorize :)
# Example: colorize([1,2,1,4]) == ['red', 'blue', 'red', 'pink']
# 
def random_colorize(arr):
    unique_arr = list(np.unique(arr))
    n_unique = len(unique_arr)
    available_colors = list(cnames.keys())
    n_av_cols = len(available_colors)
    
    if n_av_cols < n_unique:
        print('Too big array. Please, use another function.')
        return None

    colors = np.random.choice(available_colors, size=n_unique, replace=False)
    
    color_array = []
    for elem in arr:
        color_array.append(colors[unique_arr.index(elem)])
    return color_array

def colorize(arr, rnge=(0.2,0.4)):
    assert rnge[0] <= 1 and rnge[0] >= 0
    assert rnge[1] <= 1 and rnge[1] >= 0
    assert rnge[0] <= rnge[1]
    
    unique_arr = list(np.unique(arr))
    n_unique = len(unique_arr)
    available_colors = list(cnames.keys())
    n_av_cols = len(available_colors)
    available_colors = available_colors[int(round(rnge[0]*n_av_cols)): int(round(rnge[1]*n_av_cols))]
    n_av_cols = len(available_colors)
    
    if n_av_cols < n_unique:
        print('Too big array. Please, use another function.')
        return None
    
    colors = np.array(available_colors)[[(i*n_av_cols//n_unique) for i in range(n_unique)]]
    
    color_array = []
    for elem in arr:
        color_array.append(colors[unique_arr.index(elem)])
    return color_array


# To test any colorization function you want
# 
def test_colorize(N=20, colorize_function = colorize):
    x = np.linspace(0,1,N)
    plt.figure(figsize=(10,2))
    plt.title('Colorization test of 3 similar samples',fontsize=15)
    plt.scatter(x,np.ones_like(x)+3,c = colorize_function(x))
    plt.scatter(x,np.ones_like(x)+2,c = colorize_function(x))
    plt.scatter(x,np.ones_like(x)+1,c = colorize_function(x))
    plt.show()
    

# Plots ROC curve. 
# 
def ROC_plot(ax, model, _X_test, _y_test, _X_train, _y_train):
    y_score = lambda XX: model.predict(XX).ravel()
    
    fpr_test, tpr_test, threshold_test = metrics.roc_curve(_y_test, y_score(_X_test))
    roc_auc_test = metrics.auc(fpr_test, tpr_test)
    
    fpr_train, tpr_train, threshold_train = metrics.roc_curve(_y_train, y_score(_X_train))
    roc_auc_train = metrics.auc(fpr_train, tpr_train)
    
    ax.set_title('Receiver Operating Characteristic', fontsize=15)
    ax.plot([0, 1], [0, 1],'r--')
    ax.set_xlim([-0.01, 1.01])
    ax.set_ylim([-0.01, 1.01])
    ax.set_ylabel('True Positive Rate', fontsize=13) 
    ax.set_xlabel('False Positive Rate', fontsize=13)
    ax.plot(fpr_test, tpr_test, 'b', label = 'test AUC = %0.2f' % roc_auc_test)
    ax.plot(fpr_train, tpr_train, 'g', label = 'train AUC = %0.2f' % roc_auc_train)
    ax.legend(loc = 'lower right')
    

# Plots loss using array with loss values.
# To obtain losses array:
#   history = model.fit(...)
#   losses += history.history['loss']
# 
def loss_plot(ax, losses, square=1):
    ax.set_title('Loss evolution', fontsize=15)
    ax.set_ylabel('Loss', fontsize=13) 
    ax.set_xlabel('Epoch', fontsize=13)
    ax.scatter(np.arange(0,len(losses),1), losses, s=square, alpha=1, c='black')


# Plots train and test accuracy evolution.
# tst_accs, trn_accs - arrays with test and train accuracies respectively (y)
# points - array with points of measure (x)
# 
def train_test_acc_plot(ax, tst_accs, trn_accs, points):    
    ax.set_title('Accuracies', fontsize=15)
    ax.set_ylabel('Accuracy', fontsize=13) 
    ax.set_xlabel('Epoch', fontsize=13)
    ax.plot(points, tst_accs, 'b', label = 'Test')
    ax.plot(points, trn_accs, 'g', label = 'Train')
    ax.legend(loc = 'lower right')
    

# Plots a 2-d map with the results of 2-d classification with the dataset.
# wideness = 0.7 means the plot area would be scaled in 70 perc in both (x and y) directions
# poly is a current PolynomialFeatures
# 
def pred_plot(ax, model, X_sample, y_true, poly='', is_upscaled=False, wideness=0.1):
    feature_names = X_sample.keys()
    n_features = len(feature_names)
    
    if n_features > 2: 
        print('ERROR')
        return None
    
    nn = 1000  # Density
    borders_dict = {feature: (min(X_sample[feature]) - (max(X_sample[feature])-min(X_sample[feature]))*wideness, 
                          max(X_sample[feature]) + (max(X_sample[feature])-min(X_sample[feature]))*wideness
                         ) for feature in feature_names}
    nn = 4
    xx1, yy1 = np.meshgrid(*(np.linspace(borders_dict[feature][0], 
                                         borders_dict[feature][1], 
                                         nn) for feature in feature_names ) )
    zz = np.concatenate([xx1[:,...,None],yy1[:,...,None]],axis=2)
    zz = zz.reshape(np.shape(xx1)[0]*np.shape(xx1)[1],2)
    if is_upscaled:
        zz = poly.fit_transform(zz)
    z = model.predict(zz).reshape(np.shape(xx1)[0],np.shape(xx1)[1])
    h = ax.contourf(xx1,yy1,z)
    bin_colorize = lambda n: 'green' if n == 1 else 'black'
    ax.set_title('Classification map', fontsize=15)
    ax.set_ylabel('y', fontsize=13) 
    ax.set_xlabel('x', fontsize=13)
    ax.scatter(X_sample[feature_names[0]], X_sample[feature_names[1]], c=np.vectorize(bin_colorize)(y_true), alpha=0.9 )
    
# END