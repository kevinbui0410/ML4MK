# Xac dinh bai toan: classification hay regression
# chon tat ca cac thuat toan co the giai, kết hợp với cross_validation sử dụng kfold để tìm ra thuật toán tốt nhất cho dữ liệu

# inputs: X, y, CV default 10, kn_neighbors=6, forest_estimators=30, svc_kernel='linear', classification=1
# output: dataset, containt model name and accuracy when execute each model

def select_models(X, y, CV=10, kn_neighbors=6, forest_estimators=30, svc_kernel='linear', classification=True):
  
  import pandas as pd
  import numpy as np
  import warnings
  warnings.filterwarnings('ignore')
  from sklearn.model_selection import KFold
  from sklearn.model_selection import cross_val_score
  from sklearn.model_selection import train_test_split

  if classification:

    from sklearn.linear_model import LogisticRegression
    from sklearn.naive_bayes import GaussianNB
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import SVC
    from sklearn.neighbors import KNeighborsClassifier
    
    models = [
            KNeighborsClassifier(n_neighbors=kn_neighbors),
            DecisionTreeClassifier(),
            RandomForestClassifier(n_estimators=forest_estimators),
            SVC(kernel=svc_kernel),
            GaussianNB(),
            LogisticRegression(solver='lbfgs', multi_class='auto')]
  else:

    from sklearn.linear_model import LinearRegression
    from sklearn.naive_bayes import GaussianNB
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.svm import SVR
    from sklearn.neighbors import KNeighborsRegressor
    
    models = [
            KNeighborsRegressor(n_neighbors=kn_neighbors),
            DecisionTreeRegressor(),
            RandomForestRegressor(n_estimators=forest_estimators),
            SVR(kernel=svc_kernel),
            GaussianNB(),
            LinearRegression()]
  #CV = 10 # so lan lap de chay model
  entries = [] # luu ket qua 6 model, moi model 10 gia tri
  kfold = KFold(n_splits=CV)
  for model in models:
    model_name = model.__class__.__name__ # lay ten class
    accuracies = cross_val_score(model, X, y, scoring='accuracy', cv=kfold)
    #print(accuracies)
    entries.append([model_name, accuracies.mean()])

  cv_df = pd.DataFrame(entries, columns=['model_name','accuracy'])
  return cv_df

# Hàm tìm ra tỷ lệ trộn tối ưu cho 1 model cụ thể

# inputs: model name, X, y, list test size default value=[0.3,0.25,0.2], number of test default value = 10
# ouput: dataframe có columns là tổng số lần thực hiện, index là tỷ lệ trộn, mỗi cell là bộ 3 score train, score test và abs(score_train-score_test)

def select_test_size_split(model, X, y, lst_size=[0.3,0.25,0.2], num_of_test=10):

  import pandas as pd
  import numpy as np
  import warnings
  warnings.filterwarnings('ignore')
  from sklearn.model_selection import train_test_split

  dic_test_t = {}
  dic_train_t = {}
  dic_abs_t = {}
  for t in range(num_of_test):
    
    test_size_list = lst_size#[0.3,0.25,0.2]
    dic_train = {}
    dic_test = {}
    dic_abs = {}
    for i in test_size_list:

      X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=i)
      #clf1 = RandomForestClassifier(n_estimators=100)
      model.fit(X_train,y_train)

      score_train = round(model.score(X_train,y_train),2)
      score_test = round(model.score(X_test,y_test),2)
      abs_score = round(abs(score_train-score_test),2)
      dic_train[i] = score_train
      dic_test[i] = score_test
      dic_abs[i] = abs_score
      # print('With [',1-i,':',i,'] score train is ',round(score_train,2),' score test is ',round(score_test,2), 'diff if ',round(abs(score_train-score_test),2))
    dic_test_t[t] = dic_test
    dic_train_t[t] = dic_train
    dic_abs_t[t] = dic_abs
  df_train = pd.DataFrame(dic_train_t)
  df_test = pd.DataFrame(dic_test_t)
  df_abs = pd.DataFrame(dic_abs_t)
  df = pd.concat([df_train,df_test,df_abs], keys=['train', 'test', 'abs'])
  return df