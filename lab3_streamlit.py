import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split,KFold,cross_val_score,GridSearchCV
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.feature_selection import VarianceThreshold,SequentialFeatureSelector
from sklearn.decomposition import FactorAnalysis
from sklearn.linear_model import LinearRegression,LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score,accuracy_score,precision_score,recall_score,f1_score

st.set_page_config(page_title="Spotify ML Feature Selection",layout="wide")
st.title("Spotify Feature Selection and Regression Analysis")
file=st.file_uploader("Upload SpotifyAudioFeaturesApril2019.csv",type=["csv"])

if file is None:
    st.info("Please upload the Spotify CSV file.")
    st.stop()

df=pd.read_csv(file)
st.subheader("1. Dataset Exploration")
st.write("Instances:",df.shape[0],"Features:",df.shape[1])
st.dataframe(df.head())
st.write("Missing Values")
st.dataframe(df.isnull().sum())

features=['acousticness','danceability','duration_ms','energy','instrumentalness','key','liveness','loudness','mode','speechiness','tempo','time_signature','valence']
X=df[features].copy()
y_reg=df['popularity'].copy()
median_popularity=y_reg.median()
y_cls=(y_reg>=median_popularity).astype(int)

X_train,X_test,y_reg_train,y_reg_test,y_cls_train,y_cls_test=train_test_split(X,y_reg,y_cls,test_size=0.20,random_state=42,stratify=y_cls)
imputer=SimpleImputer(strategy='median')
X_train_imp=imputer.fit_transform(X_train)
X_test_imp=imputer.transform(X_test)
scaler=StandardScaler()
X_train_scaled=scaler.fit_transform(X_train_imp)
X_test_scaled=scaler.transform(X_test_imp)
X_train_scaled_df=pd.DataFrame(X_train_scaled,columns=features)
X_test_scaled_df=pd.DataFrame(X_test_scaled,columns=features)

st.subheader("2. Low Variance Filter")
mm=MinMaxScaler()
X_minmax=mm.fit_transform(X_train_imp)
vt=VarianceThreshold(threshold=0.01)
vt.fit(X_minmax)
low_features=[f for f,k in zip(features,vt.get_support()) if k]
X_train_low=X_train_scaled[:,vt.get_support()]
X_test_low=X_test_scaled[:,vt.get_support()]
st.write("Selected Features:",low_features)

st.subheader("3. High Correlation Filter")
corr=X_train_scaled_df.corr()
fig,ax=plt.subplots(figsize=(10,7))
ax.imshow(corr,cmap='coolwarm',aspect='auto')
ax.set_xticks(range(len(features)))
ax.set_xticklabels(features,rotation=90)
ax.set_yticks(range(len(features)))
ax.set_yticklabels(features)
plt.colorbar(ax.images[0])
st.pyplot(fig)
upper=corr.where(np.triu(np.ones(corr.shape),k=1).astype(bool))
high_drop=[c for c in upper.columns if any(abs(upper[c])>0.90)]
high_features=[f for f in features if f not in high_drop]
X_train_high=X_train_scaled_df[high_features].values
X_test_high=X_test_scaled_df[high_features].values
st.write("Selected Features:",high_features)

st.subheader("4. Factor Analysis")
fa=FactorAnalysis(n_components=5,random_state=42)
X_train_fa=fa.fit_transform(X_train_scaled)
X_test_fa=fa.transform(X_test_scaled)
st.write("Number of Factors:",X_train_fa.shape[1])

st.subheader("5. Backward Feature Elimination")
n=min(10000,len(X_train_scaled))
idx=np.random.RandomState(42).choice(len(X_train_scaled),n,replace=False)
X_sample=X_train_scaled[idx]
y_sample=y_reg_train.iloc[idx]
lr=LinearRegression()
backward=SequentialFeatureSelector(lr,n_features_to_select=6,direction='backward',scoring='neg_mean_squared_error',cv=3)
backward.fit(X_sample,y_sample)
backward_features=[f for f,k in zip(features,backward.get_support()) if k]
X_train_back=X_train_scaled_df[backward_features].values
X_test_back=X_test_scaled_df[backward_features].values
st.write("Selected Features:",backward_features)

st.subheader("6. Forward Feature Selection")
forward=SequentialFeatureSelector(lr,n_features_to_select=6,direction='forward',scoring='neg_mean_squared_error',cv=3)
forward.fit(X_sample,y_sample)
forward_features=[f for f,k in zip(features,forward.get_support()) if k]
X_train_for=X_train_scaled_df[forward_features].values
X_test_for=X_test_scaled_df[forward_features].values
st.write("Selected Features:",forward_features)

feature_sets={'Original':(X_train_scaled,X_test_scaled),'Low Variance':(X_train_low,X_test_low),'High Correlation':(X_train_high,X_test_high),'Backward':(X_train_back,X_test_back),'Forward':(X_train_for,X_test_for),'Factor Analysis':(X_train_fa,X_test_fa)}

st.subheader("7. Selected Feature Summary")
summary=pd.DataFrame({"Feature Set":list(feature_sets.keys()),"Number of Features":[x[0].shape[1] for x in feature_sets.values()]})
st.dataframe(summary)

st.subheader("8. Linear and Logistic Regression")
results=[]
for name,(a,b) in feature_sets.items():
    lm=LinearRegression().fit(a,y_reg_train)
    lp=lm.predict(b)
    log=LogisticRegression(max_iter=1000).fit(a,y_cls_train)
    cp=log.predict(b)
    results.append([name,np.sqrt(mean_squared_error(y_reg_test,lp)),mean_absolute_error(y_reg_test,lp),r2_score(y_reg_test,lp),accuracy_score(y_cls_test,cp),precision_score(y_cls_test,cp),recall_score(y_cls_test,cp),f1_score(y_cls_test,cp)])

results_df=pd.DataFrame(results,columns=["Feature Set","RMSE","MAE","R2","Accuracy","Precision","Recall","F1"])
st.dataframe(results_df)

st.subheader("9. Linear Regression Gradient Descent")
def linear_gd(X,y,lr=0.01,iterations=1000):
    X=np.c_[np.ones(X.shape[0]),X]
    y=(y-y.mean())/y.std()
    w=np.zeros(X.shape[1])
    loss=[]
    for i in range(iterations):
        pred=X@w
        err=pred-y
        w-=lr*(X.T@err)/len(y)
        loss.append(np.mean(err**2))
    return w,loss

rates=[0.001,0.01,0.1]
fig,ax=plt.subplots()
for r in rates:
    _,loss=linear_gd(X_train_scaled,y_reg_train,r,500)
    ax.plot(loss,label=f"LR={r}")
ax.set_xlabel("Iterations")
ax.set_ylabel("Loss")
ax.set_title("Learning Rate vs Convergence")
ax.legend()
st.pyplot(fig)

st.subheader("10. Logistic Regression Gradient Descent")
def logistic_gd(X,y,lr=0.01,iterations=1000):
    X=np.c_[np.ones(X.shape[0]),X]
    w=np.zeros(X.shape[1])
    loss=[]
    for i in range(iterations):
        z=np.clip(X@w,-500,500)
        p=1/(1+np.exp(-z))
        err=p-y
        w-=lr*(X.T@err)/len(y)
        loss.append(-np.mean(y*np.log(p+1e-9)+(1-y)*np.log(1-p+1e-9)))
    return w,loss

_,log_loss=logistic_gd(X_train_scaled,y_cls_train,0.01,500)
fig,ax=plt.subplots()
ax.plot(log_loss)
ax.set_xlabel("Iterations")
ax.set_ylabel("Loss")
ax.set_title("Logistic Regression Convergence")
st.pyplot(fig)

st.subheader("11. Effect of Iterations")
iters=[100,500,1000,2000]
final_loss=[]
for i in iters:
    _,loss=linear_gd(X_train_scaled,y_reg_train,0.01,i)
    final_loss.append(loss[-1])
fig,ax=plt.subplots()
ax.plot(iters,final_loss,marker='o')
ax.set_xlabel("Iterations")
ax.set_ylabel("Final Loss")
ax.set_title("Iterations vs Loss")
st.pyplot(fig)

st.subheader("12. 5-Fold Cross Validation")
cv=KFold(n_splits=5,shuffle=True,random_state=42)
cv_results=[]
for name,(a,b) in feature_sets.items():
    scores=np.sqrt(-cross_val_score(LinearRegression(),a,y_reg_train,cv=cv,scoring='neg_mean_squared_error'))
    cv_results.append([name,scores.mean(),scores.std()])
cv_df=pd.DataFrame(cv_results,columns=["Feature Set","Mean RMSE","Std RMSE"])
st.dataframe(cv_df)

st.subheader("13. GridSearchCV - Linear Regression")
grid_lr=GridSearchCV(LinearRegression(),{'fit_intercept':[True,False]},cv=5,scoring='neg_mean_squared_error')
grid_lr.fit(X_train_scaled,y_reg_train)
st.write("Best Parameters:",grid_lr.best_params_)
st.write("Best CV RMSE:",np.sqrt(-grid_lr.best_score_))

st.subheader("14. GridSearchCV - Logistic Regression")
grid_log=GridSearchCV(LogisticRegression(max_iter=1000),{'C':[0.01,0.1,1,10],'solver':['liblinear']},cv=5,scoring='accuracy')
grid_log.fit(X_train_scaled,y_cls_train)
st.write("Best Parameters:",grid_log.best_params_)
st.write("Best CV Accuracy:",grid_log.best_score_)

st.subheader("15. Model Comparison")
fig,ax=plt.subplots(figsize=(9,5))
ax.bar(results_df["Feature Set"],results_df["RMSE"])
ax.set_ylabel("RMSE")
ax.set_title("RMSE Comparison")
plt.xticks(rotation=30)
st.pyplot(fig)

fig,ax=plt.subplots(figsize=(9,5))
ax.bar(results_df["Feature Set"],results_df["Accuracy"])
ax.set_ylabel("Accuracy")
ax.set_title("Accuracy Comparison")
plt.xticks(rotation=30)
st.pyplot(fig)

st.subheader("16. Self-Learning - Random Forest")
rf=RandomForestRegressor(n_estimators=100,max_depth=10,random_state=42,n_jobs=-1)
rf.fit(X_train_scaled,y_reg_train)
rf_pred=rf.predict(X_test_scaled)
rf_rmse=np.sqrt(mean_squared_error(y_reg_test,rf_pred))
lr_pred=LinearRegression().fit(X_train_scaled,y_reg_train).predict(X_test_scaled)
lr_rmse=np.sqrt(mean_squared_error(y_reg_test,lr_pred))
st.write("Linear Regression RMSE:",lr_rmse)
st.write("Random Forest RMSE:",rf_rmse)
st.subheader("17. Final Results")
st.dataframe(results_df)
st.write("Cross Validation Results")
st.dataframe(cv_df)
st.write("Best Linear Parameters:",grid_lr.best_params_)
st.write("Best Logistic Parameters:",grid_log.best_params_)
st.write("Linear Regression RMSE:",lr_rmse)
st.write("Random Forest RMSE:",rf_rmse)
st.success("Analysis completed successfully.")