###  Purchase timing
###  transfer string month to date and then pick month from it as integer
pd.to_datetime(model_data.first_date_month).dt.month.astype('int')

###  create categorical variable through bins 
pd.cut(model_data['age'],bins=[0,19,35,float('Inf')],labels=['kids','adult','family'])

### transfer categorical into 1 and 0 in mannual way
np.where(model_data['age_range']=='adult',1,0).astype('int')

### select columns based on types
data.select_dtypes(exclude=['object','category'])

### identify the quantile value of a series
data[i].quantile(pct)

### create dummy variable
pd.get_dummies(model_data[['age_range','gender']])

### select both index and column in a dataframe
model_data.loc[model_feature.index,'unit_purchase']

### dictionary of models 
model_dict = {
    'logistic': LogisticRegression(),
    'LDA': LinearDiscriminantAnalysis()}

### loop through dictionary to create another dict
score_dict1 = {}
for name, model in iter(model_dict.items()):
    score_dict1[name] = cross_val_score(model, model_feature, y=model_target,scoring=make_scorer(top_lift.top_percentile_lift,needs_proba=True), n_jobs=-1, cv=3)

###  draw a ROC curve 
rf_roc_auc = roc_auc_score(model_target, forest_v.predict(model_feature))
fpr, tpr, thresholds = roc_curve(model_target, forest_v.predict_proba(model_feature)[:,1])
plt.figure()
plt.plot(fpr, tpr, label='Random Forest (area = %0.2f)' % rf_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.savefig('Log_ROC')
plt.show()

### merge two table based on both index 
pd.merge(pred_table,pd.DataFrame(pred_target,index=pred_feature.index),left_index=True, right_index=True)

### qcut to create percentile rank 
pd.qcut(pred_table['unit'],q=10,labels=False,duplicates='drop')

### apply different function to different columns after group by 
pred_table.groupby(by='rank').agg({"unit": 'mean',"unit_purchase": 'mean',"rank": 'count'})

### draw tree 
tree_data=tree.export_graphviz(tree_fit,out_file=None,feature_names=model_feature.columns)
graph=graphviz.Source(tree_data)
graph.render("explore_tree")

### use RFE to select top variables 
logreg = LogisticRegression()
rfe = RFE(logreg, 15)
log_fit=rfe.fit(model_feature,model_target)
select_cols=model_feature.columns[log_fit.support_].tolist()

### logistic regression accuracy 
logreg.score(X_test, y_test)

### pickle model save and load
filename = "rfmodel_purchasetiming.sav"
pickle.dump(forest_v1, open(filename,'wb'))
load_model=pickle.load(open(filename,'rb'))

### create new index 
first_try.set_index('device_id_hex',inplace=True)

### define new score for sklearn
import pandas as pd
import numpy as np
from sklearn.metrics import make_scorer

def top_percentile_lift(y_true, y_pred_proba): 
    y_true=pd.DataFrame(y_true,columns=['purchase'],index=y_true.index)
    y_pred_proba=pd.DataFrame(y_pred_proba,columns=['yes'],index=y_true.index)
    y_all=pd.merge(y_pred_proba,y_true,right_index=True, left_index=True)
    y_all['rank']=pd.qcut(y_all['yes'],q=10,labels=False,duplicates='drop')
    y_rank=y_all.groupby(by='rank').agg({"purchase": 'mean',"yes": 'mean'})
    y_avg=y_all['purchase'].mean()
    y_top_avg=y_rank['purchase'].tail(1)
    y_top_pred=y_rank['yes'].tail(1)
    pred_lift=y_top_avg/y_avg
    pred_acu=y_top_avg/y_top_pred
    return pred_lift

cross_val_score(model, model_feature, y=model_target,scoring=make_scorer(top_lift.top_percentile_lift,needs_proba=True), n_jobs=-1, cv=3)

### create list in one line
n_estimators = [int(x) for x in np.linspace(start = 10, stop = 200, num = 10)]

### deal with date in dataframe
model_target[alldata['first_date_month'].astype('str')>='2018-09-01']

### combine feature importance with feature names via list to dataframe
pd.concat([pd.DataFrame(forest_v3.feature_importances_.tolist()),pd.DataFrame(list(model_feature))], axis=1)

### transform time to string
datetime.date.today().strftime('%Y-%m-01')

### drop column with inplace
pred_data.drop(['purchase_next_month','forecast_next_month'],axis=1, inplace=True)

from string to time
past_date=dt.strptime(past_date,'%Y-%m-01')

### identify month of the date
pd.to_datetime(model_data.first_date_month).dt.month.astype('int')

### try except else code
try:
    export_model_result(model_date, '2000000')
except Exception as e:
    print('error raised, delete sale data:'+str(e))
    drop_value="""delete from noa.forecast_sales where month_date='"""+model_date+"""';"""
    result = connection.execute(drop_value)
    result.close()
else:
    print('model finished, data updated to JEDI')
