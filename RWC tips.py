### RWC formal tips
### feature selection and engineer pipeline
class ColumnSelector(BaseEstimator, TransformerMixin):
    def __init__(self, columns=None,remove=False):
        self.columns = columns
        self.remove=remove

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        remove=self.remove
        assert isinstance(X, pd.DataFrame), "Input not pd.DataFrame"

        try:
            if remove==True:
                cols_error = list(set(self.columns) - set(X.columns))
                assert set(X.columns)>=set(self.columns),"The DataFrame does not include the columns: %s" % cols_error
                return X[set(X.columns)-set(self.columns)]
            else: return X[self.columns]
        except KeyError:
            cols_error = list(set(self.columns) - set(X.columns))
            raise KeyError("The DataFrame does not include the columns: %s" % cols_error)
			
class TypeSelector(BaseEstimator, TransformerMixin):
    def __init__(self, dtype=None):
        self.dtype = dtype

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        assert isinstance(X, pd.DataFrame)
        return pd.concat([X.select_dtypes(include=[i]) for i in self.dtype],axis=1) 
		
# numeric type transfrom (bool, percentage, value)
class Numerictransformer(BaseEstimator,TransformerMixin):
    def __init__(self,num_bool=2,num_agg=5,skew_mark=2,out_pct=99,rm_out=False,pt_tran=True, impute_stra='constant:0'):
        self.num_bool=num_bool
        self.num_agg=num_agg
        self.skew_mark=skew_mark
        self.rm_out=rm_out
        self.impute_stra=impute_stra
        self.out_pct=out_pct
        self.pt_tran=pt_tran
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        strategy,fill_value=self.impute_stra.split(":")
        fill_value=int(fill_value) if fill_value.isdigit() else None
        pt=PowerTransformer()
        imputer=SimpleImputer(strategy=strategy,fill_value=fill_value)
        for col in X.columns:
            #outlier=np.array([0]*len(X))
            col_len=len(X[col].unique())
            high= np.nanpercentile(X[col].values,self.out_pct)
            low=np.nanpercentile(X[col].values,100-self.out_pct)
            assert ((col_len <=self.num_bool and high==1 and low==0) \
                    or col_len>=self.num_agg), "potential flawed variable exist: %s" % col
            
            if col_len>=self.num_agg and X[col].mean()<1 and X[col].mean()>0 and high <=1 and low >=0:
                #outlier[((X[col]<0) & (X[col]>1)) | (X[col].isnull())]=1
                X.loc[X[col]>1,col]=1
                X.loc[X[col]<0,col]=0
            elif col_len>=self.num_agg and X[col].skew(skipna=True)>self.skew_mark and self.rm_out==True:
                X.loc[X[col]>high,col]=high
            elif col_len>=self.num_agg and X[col].skew(skipna=True)<-self.skew_mark and self.rm_out==True:
                X.loc[(X[col]<low),col]=low
            
            X[col] = imputer.fit_transform(X[col].values.reshape(-1,1))
            
            if self.pt_tran==True and ((col_len>=self.num_agg and X[col].skew(skipna=True)>self.skew_mark) \
                                  or (col_len>=self.num_agg and X[col].skew(skipna=True)<-self.skew_mark)):
                X[col]= pt.fit_transform(X[col].values.reshape(-1, 1) )
                
        return X
		
# categorical type tranformer (onehotencoder + PCA/MCA or entity embeding using Keras or Word2Vec)
class cat_transformer(BaseEstimator,TransformerMixin):
    def __init__(self,cat_pct=0.10, method='PCA', var_in=0.85):
        self.cat_pct=cat_pct
        self.method=method
        self.var_in=var_in
    
    def fit(self, X,y=None):
        return self
    
    def transform(self, X):
        for col in X.columns:
            temp1=X[col].value_counts(dropna=False).reset_index()
            temp1['pct']=temp1[col]/temp1[col].sum()
            temp2=temp1.loc[temp1['pct']<self.cat_pct,'index'].tolist()
            X[col].replace(temp2,'small_cat',inplace=True)
        
        ohe=OneHotEncoder()
        X=pd.DataFrame.sparse.from_spmatrix(ohe.fit_transform(X))
        X.columns=ohe.get_feature_names()
        
        if self.method=='PCA':
            pca=PCA()
            pca.fit(X)
            cumsum=np.cumsum(pca.explained_variance_ratio_)
            d=np.argmax(cumsum>=self.var_in)+1
            pca=PCA(n_components=d)
            x_trans=pca.fit_transform(X)
            x_trans=pd.DataFrame(x_trans,columns=["cat_"+str(i) for i in range(d)])
        else : x_trans=X.copy()
        return x_trans
    
num_pipeline=Pipeline([
    ('typeselect',TypeSelector(['float64','int64'])),
    ('numbertransform',Numerictransformer(rm_out=False, pt_tran=False,skew_mark=2,impute_stra='median:None'))
])

cat_pipeline=Pipeline([
    ('typeselect',TypeSelector(['object'])),
    ('cattransform',cat_transformer(cat_pct=0.1,var_in=0.85,method='None')),
])

data_pipeline=Pipeline([
    ('columnselect',ColumnSelector(['naid','time','owns_switch'],remove=True)),
    ('featu',FeatureUnion([
        ('num_pipeline',num_pipeline),
        ('cat_pipeline',cat_pipeline),
    ]))
]) 


### make sure the type is correct 
isinstance(X, pd.DataFrame)

### find elements of  difference between lists
list(set(self.columns) - set(X.columns))

### identify two elements from a string
strategy,fill_value=self.impute_stra.split(":")

### if else in one line to identify digit 
fill_value=int(fill_value) if fill_value.isdigit() else None

### transform panda to numpy object for other np function
np.nanpercentile(X[col].values,self.out_pct)

### reshape the data when there is only single feature 
X[col].values.reshape(-1,1)

### value counts for categorical variables 
X[col].value_counts(dropna=False).reset_index()

### replace some values in categorical columns with other value 
X[col].replace(temp2,'small_cat',inplace=True)

### cumsum function 
cumsum=np.cumsum(pca.explained_variance_ratio_)

### argmax with cumsum to find a point 
np.argmax(cumsum>=self.var_in)+1

### create list in a loop
["cat_"+str(i) for i in range(d)]

### create customed CV split 
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

### create special list of values for search 
rvs = np.sort(loguniform.rvs(1000,len(X_train)*0.65, size=50).round().astype('int'))

### create dataframe based on several list 
LC=pd.DataFrame({'size':train_sizes,
                'train_score':train_scores.mean(axis=1),
                'test_score':validation_scores.mean(axis=1)})


### use learning curve to identify minimum sameple size 
train_sizes, train_scores, validation_scores=learning_curve(pipeline_two,X_train,y_train,train_sizes=rvs, cv=3,n_jobs=-1,scoring='roc_auc_ovr_weighted')

### create best parameter model 
rf_grid.best_estimator_
