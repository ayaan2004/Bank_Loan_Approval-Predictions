from pathlib import Path
import pandas as pd
import streamlit as st
import plotly.express as px
import joblib
from model import FEATURES, make_data, train

st.set_page_config(page_title='LoanLens | Banking Analytics', page_icon='🏦', layout='wide')
st.markdown('''<style>
.stApp {background: #f5f7fb; color:#15233c;}
[data-testid="stMetric"] {background:white; padding:17px; border-radius:12px; border:1px solid #e1e7f0;}
h1,h2,h3 {color:#142d50!important;}
[data-testid="stSidebar"] {background:#eaf0f9;}
</style>''', unsafe_allow_html=True)

@st.cache_resource
def get_artifact():
    path = Path(__file__).resolve().parent / 'loan_model.joblib'
    if path.exists():
        try:
            obj = joblib.load(path)  # Only load our own trusted local model.
            if obj.get('features') == FEATURES and obj.get('synthetic') is True:
                return obj
        except (ValueError, KeyError, EOFError, AttributeError):
            pass
    return train()

@st.cache_data
def get_data():
    return make_data()

artifact = get_artifact()
model = artifact['model']
df = get_data()
st.sidebar.title('🏦 LoanLens')
st.sidebar.caption('Bank Loan Analytics · Demo')
page = st.sidebar.radio('Navigation', ['Overview', 'Loan Predictor', 'Model Insights'])
st.sidebar.divider()
st.sidebar.warning('Educational demonstration only. All training and analytics data are synthetic. Never use for real lending decisions.')

if page == 'Overview':
    st.title('Bank Loan Approval Dashboard')
    st.caption('Interactive overview of synthetic loan applications')
    a,b,c,d = st.columns(4)
    a.metric('Applications', f'{len(df):,}')
    b.metric('Synthetic Approval Rate', f'{df.approved.mean():.1%}')
    c.metric('Average Loan Requested', f'₹{df.loan_amount.mean()/100000:.1f}L')
    d.metric('Model Test ROC AUC', f"{artifact['metrics']['roc_auc']:.3f}")
    left,right = st.columns(2)
    with left:
        status = df.approved.map({0:'Not Approved',1:'Approved'}).value_counts().reset_index()
        status.columns = ['Status','Count']
        st.plotly_chart(px.pie(status, names='Status', values='Count', hole=.58,
                               title='Synthetic Approval Distribution', color='Status',
                               color_discrete_map={'Approved':'#0d9488','Not Approved':'#e27b60'}), use_container_width=True)
    with right:
        st.plotly_chart(px.histogram(df, x='credit_score', color=df.approved.map({0:'Not Approved',1:'Approved'}),
                                       nbins=28, barmode='overlay', title='Credit Score Distribution',
                                       labels={'color':'Outcome','credit_score':'Credit score'}), use_container_width=True)
    st.subheader('Sample Applications')
    st.dataframe(df.head(25).rename(columns={'approved':'synthetic_approved'}), use_container_width=True, hide_index=True)
    st.download_button('Download synthetic demo data (CSV)', df.to_csv(index=False),
                       file_name='synthetic_loan_applications.csv', mime='text/csv')

elif page == 'Loan Predictor':
    st.title('Loan Approval Probability Simulator')
    st.caption('Try hypothetical applicant inputs. Results reflect a model trained on invented data.')
    with st.form('loan_form'):
        c1,c2 = st.columns(2)
        with c1:
            income = st.number_input('Annual Income (₹)', min_value=180000, max_value=2500000, value=800000, step=10000)
            amount = st.number_input('Requested Loan Amount (₹)', min_value=50000, max_value=3000000, value=500000, step=10000)
            term = st.selectbox('Loan Term (months)', [12,24,36,48,60,84], index=4)
            credit = st.slider('Illustrative Credit Score', 300, 900, 720)
        with c2:
            debt = st.number_input('Existing Outstanding Debt (₹)', min_value=0, max_value=1100000, value=100000, step=10000)
            years = st.slider('Employment Experience (years)', 0, 30, 5)
            late = st.slider('Late Payments in Past 12 Months', 0, 6, 0)
        submitted = st.form_submit_button('Calculate Demo Probability', type='primary', use_container_width=True)
    if submitted:
        row = pd.DataFrame([[income, amount, term, credit, debt, years, late]], columns=FEATURES)
        probability = float(model.predict_proba(row)[0,1])
        st.subheader('Simulation Result')
        l,r = st.columns([1,2])
        l.metric('Estimated Synthetic Approval Probability', f'{probability:.1%}')
        r.progress(probability)
        if probability >= 0.5:
            st.success('Model class: likely approved in this synthetic simulation (threshold 50%).')
        else:
            st.warning('Model class: unlikely approved in this synthetic simulation (threshold 50%).')
        st.info('This is not a bank approval, eligibility determination, or financial advice. Synthetic probabilities are not calibrated to real-world applicants.')
        st.write('**Input summary**')
        st.dataframe(row.T.rename(columns={0:'Value'}), use_container_width=True)

else:
    st.title('Model Insights')
    st.caption('Random Forest trained and tested on synthetic applications')
    a,b = st.columns(2)
    a.metric('Holdout Accuracy', f"{artifact['metrics']['accuracy']:.1%}")
    b.metric('Holdout ROC AUC', f"{artifact['metrics']['roc_auc']:.3f}")
    importances = pd.DataFrame({'Feature':FEATURES,'Importance':model.feature_importances_}).sort_values('Importance')
    st.plotly_chart(px.bar(importances, x='Importance', y='Feature', orientation='h',
                           title='Random Forest Feature Importance'), use_container_width=True)
    report = artifact['metrics']['classification_report']
    report_df = pd.DataFrame({k:v for k,v in report.items() if isinstance(v,dict)}).T
    st.subheader('Synthetic Holdout Classification Report')
    st.dataframe(report_df, use_container_width=True)
    st.warning('Performance metrics measure how the model reproduces an invented data-generating process. They do not demonstrate real-world accuracy, fairness, compliance, or suitability for underwriting.')
