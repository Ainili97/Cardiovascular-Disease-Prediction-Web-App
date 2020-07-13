import streamlit as st
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.externals import joblib
from math import exp

st.write("""
# Cardiovascular Disease Prediction
專屬於臺灣人心血管疾病的風險預測 :hospital:
""")

st.header('請輸入以下資訊')

def user_input_features():

    sex = st.selectbox( '性別', ['男', '女'])
    AGE = st.slider('年齡', 20, 100, 30)
    height = st.slider('身高(cm)', 140, 200, 165)
    weight = st.slider('體重(kg)', 35, 130, 60)
    BMI = weight/((height/100)**2)
    st.write('#### BMI:', round(BMI, 4))
    education = st.selectbox( '教育程度', ['國小及以下', '國中', '高中', '學士', '碩士及以上'])
    drk = st.selectbox( '飲酒狀況', ['從未喝酒', '已戒酒', '仍有飲酒習慣'])
    SMK_EXPERIENCE = st.checkbox('是否曾經吸菸')
    HYPERTENSION_SELF = st.checkbox('是否有高血壓')
    HYPERLIPIDEMIA_SELF = st.checkbox('是否有高血脂')
    APOPLEXIA_SELF = st.checkbox('是否曾中風')
    DIABETES_SELF = st.checkbox('是否有糖尿病')
    family = st.checkbox('是否有家人罹患心血管疾病')
    
    SEX = {'男': 1, '女': 0}[sex]
    EDUCATION = {'國小及以下': 3,
                 '國中': 4,
                 '高中': 5,
                 '學士': 6,
                 '碩士及以上': 7}[education]
    DRK = {'從未喝酒':1,
           '已戒酒':2,
           '仍有飲酒習慣':3}[drk]
    

    data = {'HYPERLIPIDEMIA_SELF': HYPERLIPIDEMIA_SELF,
            'HYPERTENSION_SELF': HYPERTENSION_SELF,
            'APOPLEXIA_SELF': APOPLEXIA_SELF,
            'DIABETES_SELF': DIABETES_SELF,
            'family': family,
            'AGE': AGE,
            'SEX':SEX,
            'EDUCATION': EDUCATION,
            'DRK': DRK,
            'SMK_EXPERIENCE': SMK_EXPERIENCE,
            'BMI': BMI}
    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()

#st.subheader('User Input parameters')
#st.write(df)

cb = CatBoostClassifier()
cb.load_model('save/cb.cbm')
nb = joblib.load('save/nb.pkl')
#logistic = joblib.load('save/logistic.pkl')

dummy = df.copy()
dummy['EDUCATION_4'] = [1] if dummy['EDUCATION'][0]==4 else [0]
dummy['EDUCATION_5'] = [1] if dummy['EDUCATION'][0]==5 else [0]
dummy['EDUCATION_6'] = [1] if dummy['EDUCATION'][0]==6 else [0]
dummy['EDUCATION_7'] = [1] if dummy['EDUCATION'][0]==7 else [0]
dummy['DRK_2'] = [1] if dummy['DRK'][0]==2 else [0]
dummy['DRK_3'] = [1] if dummy['DRK'][0]==3 else [0]
dummy.drop(['DRK', 'EDUCATION'], axis=1, inplace=True)

logistic = -3.67929510 + dummy.iat[0,0]*0.66939431 + dummy.iat[0,1]*0.34963816 + dummy.iat[0,2]*0.49326235 + dummy.iat[0,3]*0.28782622 + dummy.iat[0,4]*0.46924237 + dummy.iat[0,5]*0.04793672 + dummy.iat[0,6]*-0.06787666 + dummy.iat[0,7]*-0.48337344 + dummy.iat[0,8]*-0.28040232 + dummy.iat[0,9]*-0.27776647 + dummy.iat[0,10]*-0.36949343 + dummy.iat[0,11]*0.38791514 + dummy.iat[0,12]*-0.26635957 + dummy.iat[0,13]*0.28383488 + dummy.iat[0,14]*0.01441903


prediction_cb = cb.predict_proba(df)[0][1]
prediction_nb = nb.predict_proba(dummy)[0][1]
#prediction_logistic = logistic.predict_proba(dummy)[0][1]
ex = exp(logistic)
prediction_logistic = exp(logistic)/(1-exp(logistic))

prediction_proba = sum([prediction_cb, prediction_nb, prediction_logistic])/3

st.markdown('---')
st.subheader('您罹患心血管疾病的風險為：')

if prediction_proba>0.5:
    st.write('# <font style="color: #e63946; background: #fafafa">%.4f</font>'%prediction_proba,unsafe_allow_html=True)
    st.write(':cupid: 您可能為心血管疾病的高風險族群！')
else:
    st.write('# <font style="background: #fafafa">%.4f</font>'%prediction_proba,unsafe_allow_html=True)