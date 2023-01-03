import flask, pickle
from flask import Flask, request, render_template
import numpy as np
import pandas as pd

from sklearn.metrics import f1_score, average_precision_score
from sklearn.metrics import make_scorer, accuracy_score, recall_score, precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.pipeline import Pipeline
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split
from lightgbm import LGBMClassifier

app = Flask(__name__)

with open('./2. modeling/model_1.pkl','rb') as f:
    model_1 = pickle.load(f)
with open('./2. modeling/model_2.pkl','rb') as f:
    model_2 = pickle.load(f)
with open('./2. modeling/model_3.pkl','rb') as f:
    model_3 = pickle.load(f)
with open('./2. modeling/model_4.pkl','rb') as f:
    model_4 = pickle.load(f)
with open('./2. modeling/model_5.pkl','rb') as f:
    model_5 = pickle.load(f)
        
def print_proba(X_data):
    X_test = X_data.values.reshape(1,-1)
    y_pred_1 = model_1.predict_proba(X_test)[0][1]
    y_pred_2 = model_2.predict_proba(X_test)[0][1]
    y_pred_3 = model_3.predict_proba(X_test)[0][1]
    y_pred_4 = model_4.predict_proba(X_test)[0][1]
    y_pred_5 = model_5.predict_proba(X_test)[0][1]
    
    first_ment = f'해당 기업의 예상 폐업률은 \n1년 내 {np.round(y_pred_1*100, 1)}%, \n2년 내 {np.round(y_pred_2*100, 1)}%, \n3년 내 {np.round(y_pred_3*100, 1)}%, \n4년 내 {np.round(y_pred_4*100, 1)}%, \n5년 내 {np.round(y_pred_5*100,1)}% 입니다. '
    if y_pred_1 > 0.5:
        second_ment = f'1년 내 폐업이 예상되오니 1년 미만 단기계약을 포함한 모든 계약에 신중하시기 바랍니다.'
    elif y_pred_2 > 0.5:
        second_ment = f'2년 내 폐업이 예상되오니 1년 미만 단기계약을 제외한 1년 이상 장기계약에 신중하시기 바랍니다.'
    elif y_pred_3 > 0.5:
        second_ment = f'3년 내 폐업이 예상되오니 2년 이상 장기계약에 신중하시기 바랍니다.'
    elif y_pred_4 > 0.5:
        second_ment = f'4년 내 폐업이 예상되오니 3년 이상 장기계약에 신중하시기 바랍니다.'
    elif y_pred_5 > 0.5:
        second_ment = f'5년 내 폐업이 예상되오니 4년 이상 장기계약에 신중하시기 바랍니다.'
    else :
        second_ment = f'5년 내 폐업 가능성이 낮습니다. 안심하고 계약하셔도 좋습니다.'
    return first_ment, second_ment 
    

def log_the_user_in(name=None):
    print(name)
    return render_template('index2.html', name=name)

def valid_login(name=None, password=None):

    if name == 'AIB' and password == '161616':
        return True
    else:
        return False

#로그인 페이지 라우팅
app.route("/")
@app.route('/', methods=['POST', 'GET'])
def login():
    error = None
    if request.method == 'POST':
        if valid_login(request.form['username'],
                      request.form['password']):
            return log_the_user_in(request.form['username'])
        else:
            error = 'Invalid username/password'
            
    # the code below is executed if the request method
    # was GET or the credentials were invalid
    return render_template('login2.html', error=error)
    

# 데이터 예측 처리
@app.route('/result', methods=['POST', 'GET'])
def index(data=None):
    
    try : 
        data = pd.Series([request.form['fiscal_yr'],
        request.form['asset'],
        request.form['capital'],
        request.form['total_equity'],
        request.form['revenue'],
        request.form['oprt_income'],
        request.form['net_income']]).astype(int) # 컬럼 순서변경
        return render_template('answer.html',
                               first_answer = print_proba(data)[0],
                               second_answer = print_proba(data)[1])    
        
    except:    # 예외가 발생했을 때 실행됨
        return render_template('answer.html',
                               first_answer = '누락된 데이터가 있거나 입력 데이터에 공백이 있습니다.',
                               second_answer = '데이터를 다시 입력해주세요')    

    # Flask 서비스 스타트
    app.run(host='0.0.0.0', port=8080, debug=False)
