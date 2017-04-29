"""
Homework 2.

Author: Michael Jaron
uni: mj2776
"""
import pandas as pd
import numpy as np

from sklearn.linear_model import Ridge

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# import matplotlib.pyplot as plt


def score_rent(model, X_test, y_test):
	"""
	Calculate model score, R^2.
	"""
	r2 = model.score(X_test, y_test)
	return r2


def predict_rent(model, X_test, y_test):
	"""
	Return your test data, the true labels and your predicted labels.
	"""
	pred_labels = model.predict(X_test)

	return X_test, y_test, pred_labels


def find_missing(var, df):
	"""
	Parse messy layot file, only works with modified csv in repo.
	"""
	s = df[df['Source Code'] != 'na'].index.tolist()
	v = df[df['Variable Name'] != 'na'].index.tolist()
    
	if 'sc' in var:
	    var = var[2:]
	    if len(var) == 2:
	        var = '0' + var
	    a = df[df['Source Code'] == var].index.tolist()
	    loc_s = s.index(a[0])
	    num2 = s[loc_s + 1]
	    temp = v
	else:
	    a = df[df['Variable Name'] == var.upper()].index.tolist()
	    loc_v = v.index(a)
	    num2 = v[loc_v + 1]
	    temp = s

	for i in range(a[0], max(temp)):
	    try:
	        loc = temp.index(i)
	        break
	    except:
	        pass

	if temp[loc] < num2:
	    num2 = temp[loc]

	df1 = df['Code and Description'][a[0]:num2]

	missing = []
	for i in df1:
	    name = i[i.find('=') + 1:].lower()
	    if 'not reported' in name or 'not selected' in name or \
	    	'not applicable' in name or 'not available' in name or \
	    	'not found' in name:
	        missing.append(i[:i.find('=')])

	return missing


def main():
	seed = 7
	data_url = "https://ndownloader.figshare.com/files/7586326"
	df = pd.read_csv(data_url)

	# all features that have to do with renters
	remove = ['uf31', 'SEQNO', 'UF17A', 'REC1', 'REC6', 'UF46', 'REC4', 
	'Rec_Race_A', 'Rec_Race_C', 'REC28', 'PPR', 'TOT_PER', 'REC35', 'UF26', 
	'REC37', 'UF28', 'REC33', 'UF27', 'REC39', 'REC9', 'UF42', 'UF42A', 'UF34', 
	'UF34A', 'HFLAG18', 'HFLAG4', 'HFLAG12', 'HFLAG11', 'HFLAG91', 'HFLAG10', 
	'HFLAG9', 'HFLAG2', 'HFLAG1', 'FLG_RC1', 'FLG_AG1', 'FLG_SX1', 'CHUFW', 'FW', 
	'REC7', 'REC8', 'UF29', 'REC32', 'UF30', 'REC36', 'UF40A', 'UF40', 'UF39A', 
	'UF39', 'UF38A', 'UF38', 'UF37A', 'UF37', 'UF36A', 'UF36', 'UF35A', 'UF35', 
	'RACE1', 'UF10', 'UF8', 'UF9', 'UF7a', 'UF7', 'UF6', 'Uf5', 'recid', 
	'SEX/HHR2', 'HHR2', 'HHR3T', 'UF43', 'HSPANIC/HHR5', 'HHR5', 'race1', 'UF2A', 
	'UF2ACNT', 'UF66', 'UF53', 'UF54', 'UF2B,', 'UF2b', 'UF2BCNT', 'UF2A,', 
	'UF12', 'UF13', 'UF14', 'UF15', 'UF16', 'UF64', 'sc51', 'sc52', 'sc53', 
	'sc54', 'sc110', 'sc111', 'sc112', 'sc113', 'sc115', 'sc116', 'sc120', 
	'sc121', 'sc124', 'sc125', 'sc127', 'sc128', 'sc134', 'sc135', 'sc130', 
	'sc140', 'sc141', 'sc142', 'sc143', 'sc144', 'sc145', 'sc181', 'sc570', 
	'sc574', 'sc560', 'sc117', 'sc118', 'SC26', 'SC27', 'SC166', 'sc164', 'sc161',
	'sc159', 'sc51']

	# make them all lower case like the data
	remove = [i.lower() for i in remove]

	for i in df.keys():
	    if 'uf1_' in i or 'uf52h_' in i or i in remove:
	        df = df.drop(i, axis=1)

	# find all na's in the layout file
	layout_file = pd.read_csv('./occ_14_long.csv')
	layout_file = layout_file.fillna('na')
	
	# make a copy and get rid of any missing values
	df1 = df.copy()
	for i in df1.keys():
	    if 'boro' not in i:
	        na_val = find_missing(i, layout_file)
	        if len(na_val) > 0:
	            for num in na_val:
	                df1[i] = df1[i].replace(num, np.nan)

	# dont want to predict on missing values
	df1 = df1[df1['uf17'] != 99999]
	df1 = df1[df1['uf17'] != 7999]

	# continuious features
	cont_feat = ['sc150', 'sc151', 'uf11', 'uf48', 'uf11', 'sc571', 'rec54', 
	'rec53', 'sc186', 'uf23'] 

	y = np.array(df1['uf17'])
	X = df1.drop('uf17', axis=1)

	# fill in missing values with median
	for col in X.keys():
	    fill = round(np.mean(X[col]), 0)
	    X[col] = X[col].fillna(fill)

	# split by feature types
	x_cont = X[cont_feat]
	x_cat = X.drop(cont_feat, axis=1)

	# fix categorical variables
	one_h = OneHotEncoder()
	new_x = one_h.fit_transform(x_cat).toarray()

	# combine the two arrays
	X = np.hstack((new_x, x_cont))

	# split the data
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, 
		random_state=seed)
	
	# found that ridge works best over all linear models, and predifined
	# parameters work just as good as others
	model = Ridge(random_state=seed)

	# use standard scaler in pipeline to avoid leakage
	pipe = make_pipeline(StandardScaler(), model)

	# pring cv score on training data
	print(cross_val_score(pipe, X_train, y_train, cv=5).mean())

	# fit on training data and get score on test test
	pipe.fit(X_train, y_train)
	r2 = score_rent(pipe, X_test, y_test)
	print(r2)

	X_test, y_test, pred_labels = predict_rent(model, X_test, y_test)

	# fig, ax = plt.subplots()
	# ax.scatter(y_test, pred_labels)
	# ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4)
	# ax.set_xlabel('Measured')
	# ax.set_ylabel('Predicted')
	# plt.show()

	return model, X_test, y_test


if __name__ == '__main__':
	model, X_test, y_test = main()
