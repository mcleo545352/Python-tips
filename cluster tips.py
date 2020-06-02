### Player Clustering
### find colums by both types and column name
object_col=data1.columns[(data1.dtypes=="object") & (data1.columns!="na_id")]

### apply a function to each column 
data1[object_col].apply(pd.to_numeric, errors='coerce')

### create random value based for each record of dataframe
rand=np.random.randint(1,11,size=len(cluster_data))

### dataframe sort by value 
inertias_var_table.sort_values(['IMP'])

### use lambda in group by 
cluster_score.groupby('cluster').agg([lambda x: (x>mean_value).sum(),'count'])

### use pop to select one element of the list
pct_col.pop(pct_col.index('seconds_per_day'))

### filter data by regex 
alldata.filter(regex='first$')

### to use number index to identify dataframe
temp.iloc[:,range(1,17)]

### identify the index of certain column 
test_data.columns.tolist().index(col)
