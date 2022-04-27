import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import pyplot
from sklearn.preprocessing import RobustScaler, LabelEncoder
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report, plot_roc_curve, mean_squared_error, \
    mean_absolute_error
from sklearn.model_selection import train_test_split, cross_val_score, cross_validate


pd.set_option('display.max_columns', None)
pd.set_option('display.width', 170)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

df = pd.read_csv("datasets/hitters.csv")
df.columns = [str(i).upper() for i in list(df.columns)]

##################################
# Veri Ön İşleme
##################################

def grab_col_names(dataframe, cat_th=10, car_th=20):
    num_cols = [col for col in dataframe.columns if dataframe[col].dtype != "O"]
    num_but_cat = [col for col in num_cols if dataframe[col].nunique() < cat_th]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    cat_cols = [col for col in dataframe.columns if dataframe[col].dtype == "O"]
    cat_but_car = [col for col in cat_cols if dataframe[col].nunique() > car_th]
    cat_cols = [col for col in cat_cols if col not in cat_but_car]
    cat_cols = cat_cols + num_but_cat
    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')
    return num_cols, cat_cols, cat_but_car

def outlier_thresholds(dataframe, variable, q1=0.05, q3=0.95):
    quartile1 = dataframe[variable].quantile(q1)
    quartile3 = dataframe[variable].quantile(q3)
    interquartile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquartile_range
    low_limit = quartile1 - 1.5 * interquartile_range
    return low_limit, up_limit

def check_outlier(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    if dataframe[(dataframe[variable] < low_limit) | (dataframe[variable] > up_limit)].any(axis=None):
        return True
    else:
        return False

def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit


# KATEGORİK VE NUMERİK DEĞİŞKENLERİN SEÇİLMESİ
num_cols, cat_cols, cat_but_car = grab_col_names(df)

for col in num_cols:
    print(col, check_outlier(df, col))


# Hedef Değişken Analizinin Yapılması:
df.groupby(cat_cols)["SALARY"].mean()


##################################
# Feature Engineering
##################################

df['NEW_ATBAT_HITS_RATIO'] = df['HITS'] / df['ATBAT']  # 1986-1987 sezonundaki isabet yüzdesi
df['NEW_CATBAT_CHITS_RATIO'] = df['CHITS'] / df['CATBAT']  # Kariyeri boyunca isabet yüzdesi

df['NEW_TOTAL_SUCCESS'] = df['HITS'] + df['RUNS'] + df['RBI'] + df['WALKS'] + df['ASSISTS'] - df['ERRORS']

# Correlation Analysis #
def target_correlation_matrix(dataframe, corr_th=0.50, target="SALARY"):

    corr = dataframe.corr()
    corr_th = corr_th
    try:
        filter = np.abs(corr[target]) > corr_th
        corr_features = corr.columns[filter].tolist()
        sns.clustermap(dataframe[corr_features].corr(), annot=True, fmt=".2f")
        plt.show()
        return corr_features
    except:
        print("High threshold value, decrease corr_th value")

target_correlation_matrix(df, corr_th=0.55, target="SALARY")

df['New_High_Corr_Variables'] = df['CRUNS'] * df['CRBI']

df.drop(columns=['CRUNS', 'CRBI'], inplace=True)


# One-Hot Encoding:
def one_hot_encoder(dataframe, categorical_cols, drop_first=True):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe

num_cols, cat_cols, cat_but_car = grab_col_names(df)
num_cols = [col for col in num_cols if "SALARY" not in col]

df = one_hot_encoder(df, cat_cols, drop_first=True)


# Nümerik değişkenler için standartlaştırma:
scaler = RobustScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])


#############################################
# Model Kurma
#############################################
df.dropna(inplace=True)

y = df["SALARY"]
X = df.drop(["SALARY"], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=17)
reg_model = LinearRegression().fit(X_train, y_train)
reg_model.intercept_
reg_model.coef_

##########################
# Tahmin Başarısını Değerlendirme
##########################

y_pred = reg_model.predict(X_test)

# MSE Hesaplanması:
mse = mean_squared_error(y_test, y_pred)

# RMSE Hesaplanması
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

# MAE Hesaplanması
mae = mean_absolute_error(y_test, y_pred)


data = {"mae": [mae], "mse": [mse], "rmse": [rmse]}
df2 = pd.DataFrame(data)
print(df2)

