import pandas as pd
import numpy as np
# Suppress warnings from pandas
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
#plt.style.use('ggplot')
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from sklearn import metrics  
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split

from IPython.core.display import HTML
css_file = './css/resume.min.css'
styles = open(css_file,"r").read()
HTML(styles)

def createcolumn(df, cond_col, cond, group_col, num_of_new_cols):
    grouped_df = (df[df[cond_col]==cond].groupby([group_col]).size()/df.groupby([group_col]).size()).sort_values(ascending=False)

    grouped_list = grouped_df.index.tolist()
    val_list = grouped_list[:num_of_new_cols] + grouped_list[-num_of_new_cols:] 
    #tail = grouped_df.tail(num_of_new_cols).index.tolist()
    #val_list.append(tail)
    new_data = pd.DataFrame()
    for i in val_list:
        new_data[i+'_'+group_col] = [1 if x ==i else 0 for x in df[group_col]]
    return new_data

import nltk
from nltk.corpus import stopwords
import en_core_web_sm
from sklearn.feature_extraction import text

nltk.download('words')
nlp = en_core_web_sm.load()
words = set(nltk.corpus.words.words())
nltk.download('stopwords')

def cleaning(frame,col):
    """ 
    Function to clean text from a column in a data frame 
  
    This funtion removes non alphabethic characters,stop words and numerical characters and return text in lowers. 
  
    Parameters: 
    Data frame, text column 
  
    Returns: 
    values clean text from column
  
    """
    newframe=frame.copy()  
    #newframe=newframe[~newframe[col].isnull()]   
    newframe[col].fillna("No_Comment", inplace = True) #.fillna('No_comments')
    punc = ['.', ',', '"', "'", '?', '!', ':', ';', '(', ')', '[', ']', '{', '}','#',"%"]
    stop_words = text.ENGLISH_STOP_WORDS.union(punc)
    stop_words = list(stop_words)  
    newframe[col]=newframe[col].str.replace('\d+', '').str.replace('\W', ' ').str.lower().str.replace(r'\b(\w{1,3})\b', '')
    newframe['Cleantext'] = [' '.join([w for w in x.lower().split() if w not in stop_words]) for x in newframe[col].tolist()] 
    
    #content = newframe['Cleantext']#.values
    return newframe['Cleantext']

from textblob import TextBlob

def analysis(text_column, polarity_column1, subjectivity_column2):
    newdf = pd.DataFrame()
    newdf['Cleantext'] = text_column.copy()
    newdf[polarity_column1] = text_column.apply(lambda text: (TextBlob(text).polarity) if text!='no_comment' else None)
    newdf[subjectivity_column2] = text_column.apply(lambda text: (TextBlob(text).subjectivity) if text!='no_comment'else None)# else 0)
    return newdf[['Cleantext',polarity_column1,subjectivity_column2]]

from sklearn.feature_extraction.text import CountVectorizer
import collections

def wordfrequecyplot(cleantext,title):
    cv = CountVectorizer()
    words = cv.fit_transform(cleantext)
    word_freq = dict(zip(cv.get_feature_names(), np.asarray(words.sum(axis=0)).ravel()))
    word_counter = collections.Counter(word_freq)
    word_counter_df = pd.DataFrame(word_counter.most_common(15), columns = ['word', 'freq'])
    fig, ax = plt.subplots(figsize=(15,5))
    sns.barplot(x="word", y="freq", data=word_counter_df, ax=ax)
    plt.title(title)
    plt.show();
    
from sklearn.metrics import confusion_matrix
def modelfit(alg, x, Y, useTrainCV=True, cv_folds=5, early_stopping_rounds=50):
    
    if useTrainCV:
        xgb_param = alg.get_xgb_params()
        xgtrain = xgb.DMatrix(x.values, label=Y.values)
        
        cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=alg.get_params()['n_estimators'], nfold=cv_folds,
            metrics=['auc','logloss'], early_stopping_rounds=early_stopping_rounds, verbose_eval=False)
        alg.set_params(n_estimators=cvresult.shape[0])
    
    #Fit the algorithm on the data
    alg.fit(x, Y,eval_metric='auc')
        
    #Predict training set:
    dtrain_predictions = alg.predict(x)
    dtrain_predprob = alg.predict_proba(x)[:,1]
        
    #Print model report:
    print ("\nModel Report")
    print(alg)
    print ("Accuracy : %.4g" % metrics.accuracy_score(Y.values, dtrain_predictions))
    print ("AUC Score (Train): %f" % metrics.roc_auc_score(Y, dtrain_predprob))
                    
    feat_imp = pd.Series(alg.get_booster().get_fscore()).sort_values(ascending=False)
    #print(len(feat_imp))
    print(feat_imp[0:20])
    
    tn, fp, fn, tp = confusion_matrix(Y.values, dtrain_predictions).ravel()
    #Nm = C / C.astype(np.float).sum(axis=1)
    print('tn', tn/len(Y), 'fp', fp/len(Y), 'fn', fn/len(Y),'tp', tp/len(Y))
    return

def plot_grid_search_validation_curve(grid, param_range, param_to_vary, title='Validation Curve', ylim=None, xlim=None, log=None):                  
    """Plots train and cross-validation scores from a GridSearchCV instance's
    best params while varying one of those params."""

    df_cv_results_ = pd.DataFrame(grid.cv_results_)

    plt.clf()

    plt.title(title)
    plt.xlabel(param_to_vary)
    plt.ylabel('Score')

    if (ylim is None):
        plt.ylim(0.0, 1.1)
    else:
        plt.ylim(*ylim)

    if (not (xlim is None)):
        plt.xlim(*xlim)

    if log:
        plot_fn = plt.semilogx
    else:
        plot_fn = plt.plot

    plot_fn(param_range, df_cv_results_['mean_test_Accuracy'], label='Test Accuracy')
    plt.fill_between(param_range, df_cv_results_['mean_test_Accuracy'] - df_cv_results_['std_test_Accuracy'],
                     df_cv_results_['mean_test_Accuracy'] + df_cv_results_['std_test_Accuracy'], alpha=0.15,
                     color='r')
    plot_fn(param_range, df_cv_results_['mean_test_Recall'], label='Test Recall', color='b')
    plt.fill_between(param_range, df_cv_results_['mean_test_Recall'] - df_cv_results_['std_test_Recall'],
                     df_cv_results_['mean_test_Recall'] + df_cv_results_['std_test_Recall'], alpha=0.15,
                     color='b')
    
    plt.legend(loc='lower right')

    plt.show()
    
from mpl_toolkits.mplot3d import Axes3D
def plot_grid_search_3d_validation(grid, param_to_vary1, param_to_vary2, log1=None, log2=None):                  
    """Plots train and cross-validation scores from a GridSearchCV instance's
    best params while varying one of those params."""

    df = pd.DataFrame(grid.cv_results_)


    max_depth = df['param_'+param_to_vary1]
    min_child = df['param_'+param_to_vary2]
    Accuracy = df['mean_test_Accuracy']
    Recall = df['mean_test_Recall']

    fig = plt.figure(figsize=(10,7))
    ax = fig.add_subplot(projection='3d')

    sp = ax.scatter(max_depth, min_child, Accuracy, s=(2**(7*Accuracy)), c=Recall)
    cbar = plt.colorbar(sp)
    
    if log2:
        ax.set_yscale('log')
    elif log1:
        ax.set_xscale('log')
        
    cbar.set_label('Recall', rotation=0)
    plt.title("Validation 3D")
    plt.xlabel(param_to_vary1)
    plt.ylabel(param_to_vary2)
    ax.set_zlabel('Accuracy')

    plt.show()