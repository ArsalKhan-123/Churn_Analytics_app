import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.figure_factory as ff
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc

# --- Page Configuration ---
st.set_page_config(
    page_title="Customer Churn & Retention Analytics",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS Styling & Hero Banner ---
st.markdown("""
<style>
    /* Main Background & Base Styling */
    .stApp {
        background-color: #0E1117;
    }
    
    /* Hero Header Container */
    .hero-container {
        background: linear-gradient(135deg, #161B22 0%, #0D1117 100%);
        border: 1px solid #30363D;
        border-left: 5px solid #2F81F7;
        border-radius: 12px;
        padding: 24px 28px;
        margin-bottom: 25px;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.4);
    }
    .hero-title {
        color: #F0F6FC;
        font-size: 2.1rem;
        font-weight: 800;
        margin: 0 0 6px 0;
        letter-spacing: -0.5px;
        display: flex;
        align-items: center;
        gap: 12px;
    }
    .hero-subtitle {
        color: #8B949E;
        font-size: 1.05rem;
        margin: 0 0 16px 0;
        font-weight: 400;
    }
    /* Status Pills */
    .pill-container {
        display: flex;
        gap: 10px;
        flex-wrap: wrap;
    }
    .pill {
        background-color: #21262D;
        border: 1px solid #30363D;
        color: #C9D1D9;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.82rem;
        font-weight: 600;
        display: inline-flex;
        align-items: center;
        gap: 6px;
    }
    .pill-blue {
        border-color: #1F6FEB;
        color: #58A6FF;
    }
    .pill-green {
        border-color: #238636;
        color: #3FB950;
    }

    /* Card Metric Styling */
    div[data-testid="stMetricValue"] {
        font-size: 1.8rem !important;
        font-weight: 700 !important;
        color: #58A6FF !important;
    }
    
    div[data-testid="stMetric"] {
        background-color: #161B22;
        border: 1px solid #30363D;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
    }

    /* Tab Header Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: #161B22;
        border-radius: 6px 6px 0px 0px;
        padding: 8px 16px;
        color: #C9D1D9;
        border: 1px solid #30363D;
        border-bottom: none;
    }

    .stTabs [aria-selected="true"] {
        background-color: #1F6FEB !important;
        color: #FFFFFF !important;
        font-weight: 600;
    }
</style>

<div class="hero-container">
    <div class="hero-title">
        ⚡ Customer Churn & Retention Analytics
    </div>
    <div class="hero-subtitle">
        Enterprise Machine Learning engine for proactive churn risk prediction, ARR exposure modeling, and retention campaign targeting.
    </div>
    <div class="pill-container">
        <span class="pill pill-blue">🤖 Model: Random Forest Classifier</span>
        <span class="pill pill-green">🟢 Status: Live Inference</span>
        <span class="pill">📊 Metrics: Real-time ROI & AUC Analysis</span>
    </div>
</div>
""", unsafe_allow_html=True)

# --- Helper: Plotly Dark Layout Styling ---
def apply_dark_theme(fig):
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#C9D1D9', family='sans-serif'),
        xaxis=dict(gridcolor='#21262D', zerolinecolor='#21262D'),
        yaxis=dict(gridcolor='#21262D', zerolinecolor='#21262D'),
        margin=dict(l=20, r=20, t=40, b=20)
    )
    return fig

# --- Helper: Generate Synthetic Dataset ---
@st.cache_data
def generate_synthetic_data(n_samples=1000):
    np.random.seed(42)
    tenure = np.random.randint(1, 72, size=n_samples)
    monthly_charges = np.round(np.random.uniform(20.0, 120.0, size=n_samples), 2)
    total_charges = np.round(tenure * monthly_charges + np.random.normal(0, 50, size=n_samples), 2)
    total_charges = np.maximum(total_charges, monthly_charges)
    
    contract_type = np.random.choice(['Month-to-month', 'One year', 'Two year'], size=n_samples, p=[0.55, 0.25, 0.20])
    tech_support = np.random.choice(['Yes', 'No'], size=n_samples, p=[0.4, 0.6])
    paperless_billing = np.random.choice(['Yes', 'No'], size=n_samples, p=[0.6, 0.4])
    support_tickets = np.random.poisson(lam=2, size=n_samples)
    
    log_odds = (
        -0.05 * tenure +
        0.02 * monthly_charges +
        (contract_type == 'Month-to-month') * 1.2 -
        (contract_type == 'Two year') * 1.5 +
        (tech_support == 'No') * 0.8 +
        0.4 * support_tickets - 1.0
    )
    prob = 1 / (1 + np.exp(-log_odds))
    churn = (prob > np.random.uniform(0, 1, size=n_samples)).astype(int)
    
    return pd.DataFrame({
        'CustomerID': [f'CUST-{1000+i}' for i in range(n_samples)],
        'Tenure_Months': tenure,
        'Monthly_Charges': monthly_charges,
        'Total_Charges': total_charges,
        'Contract_Type': contract_type,
        'Tech_Support': tech_support,
        'Paperless_Billing': paperless_billing,
        'Support_Tickets_Last_6M': support_tickets,
        'Churn': churn
    })

# --- Sidebar Controls ---
st.sidebar.header("📁 Data & Model Controls")
data_source = st.sidebar.radio(
    "Select Data Source:",
    ["Use Demo Dataset", "Upload Custom CSV File"]
)

df = None

if data_source == "Use Demo Dataset":
    df = generate_synthetic_data()
    st.sidebar.success(f"Loaded demo dataset ({len(df)} records).")
else:
    uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=["csv"])
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            df.columns = df.columns.str.strip()
            if 'CustomerID' not in df.columns:
                df['CustomerID'] = [f'CUST-{1000+i}' for i in range(len(df))]
            st.sidebar.success(f"Successfully loaded file ({len(df)} rows).")
        except Exception as e:
            st.sidebar.error(f"Error reading file: {e}")
    else:
        st.info("👈 Please upload a CSV file in the sidebar to begin analysis.")

# --- Application Main Body ---
if df is not None:
    feature_cols = ['Tenure_Months', 'Monthly_Charges', 'Total_Charges', 
                    'Contract_Type', 'Tech_Support', 'Paperless_Billing', 'Support_Tickets_Last_6M']
    
    missing_cols = [col for col in feature_cols + ['Churn'] if col not in df.columns]
    if missing_cols:
        st.error(f"Uploaded CSV is missing required columns: `{missing_cols}`")
    else:
        X = df[feature_cols].copy()
        y = df['Churn']
        X_encoded = pd.get_dummies(X, drop_first=True)
        
        if len(df) >= 10:
            X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.25, random_state=42)
        else:
            X_train, X_test, y_train, y_test = X_encoded, X_encoded, y, y
            
        model = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # --- Sidebar Simulator UI ---
        st.sidebar.markdown("---")
        st.sidebar.header("🔮 Single Customer Simulator")
        
        sim_tenure = st.sidebar.slider("Tenure (Months)", 1, 72, 12)
        sim_monthly = st.sidebar.slider("Monthly Charges ($)", 20.0, 150.0, 70.0, step=5.0)
        sim_total = sim_tenure * sim_monthly
        sim_contract = st.sidebar.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
        sim_tech = st.sidebar.selectbox("Tech Support", ["No", "Yes"])
        sim_paperless = st.sidebar.selectbox("Paperless Billing", ["No", "Yes"])
        sim_tickets = st.sidebar.slider("Support Tickets (Last 6M)", 0, 10, 2)
        
        sim_dict = {
            'Tenure_Months': [sim_tenure],
            'Monthly_Charges': [sim_monthly],
            'Total_Charges': [sim_total],
            'Contract_Type': [sim_contract],
            'Tech_Support': [sim_tech],
            'Paperless_Billing': [sim_paperless],
            'Support_Tickets_Last_6M': [sim_tickets]
        }
        sim_raw_df = pd.DataFrame(sim_dict)
        sim_encoded = pd.get_dummies(sim_raw_df, drop_first=True).reindex(columns=X_encoded.columns, fill_value=0)
        sim_prob = model.predict_proba(sim_encoded)[0, 1] * 100
        
        st.sidebar.subheader("Simulation Results")
        if sim_prob >= 70:
            st.sidebar.error(f"⚠️ Risk Score: **{sim_prob:.1f}%** (High Risk)")
        elif sim_prob >= 40:
            st.sidebar.warning(f"⚡ Risk Score: **{sim_prob:.1f}%** (Medium Risk)")
        else:
            st.sidebar.success(f"✅ Risk Score: **{sim_prob:.1f}%** (Low Risk)")
            
        # Inference on full dataset
        df['Churn_Probability'] = model.predict_proba(X_encoded)[:, 1]
        df['Predicted_Churn'] = (df['Churn_Probability'] >= 0.5).astype(int)
        
        # Executive KPIs Container
        st.subheader("📌 Executive Dashboard")
        col1, col2, col3, col4 = st.columns(4)
        
        total_customers = len(df)
        predicted_churners = df[df['Predicted_Churn'] == 1]
        total_arr_at_risk = (predicted_churners['Monthly_Charges'] * 12).sum()
        avg_churn_prob = df['Churn_Probability'].mean() * 100
        
        col1.metric("Total Customers", f"{total_customers:,}")
        col2.metric("Predicted At-Risk", f"{len(predicted_churners)}", delta=f"{len(predicted_churners)/total_customers*100:.1f}%", delta_color="inverse")
        col3.metric("Annual Revenue Risk", f"${total_arr_at_risk:,.2f}")
        col4.metric("Average Churn Risk", f"{avg_churn_prob:.1f}%")
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Navigation Tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📉 Churn Risk Analysis", 
            "🧠 Feature Importance", 
            "🎯 Customer Action List",
            "💰 Financial ROI Calculator",
            "⚡ Evaluation Metrics"
        ])
        
        with tab1:
            st.subheader("Distribution & Risk Segmentation")
            c1, c2 = st.columns(2)
            
            with c1:
                with st.container(border=True):
                    fig_hist = px.histogram(
                        df, x="Churn_Probability", color="Contract_Type",
                        nbins=30, title="Churn Risk Distribution by Contract",
                        color_discrete_sequence=['#FF7F0E', '#1F77B4', '#2CA02C']
                    )
                    st.plotly_chart(apply_dark_theme(fig_hist), use_container_width=True)
                
            with c2:
                with st.container(border=True):
                    fig_scatter = px.scatter(
                        df, x="Tenure_Months", y="Monthly_Charges",
                        color="Churn_Probability", size="Support_Tickets_Last_6M",
                        title="Tenure vs. Monthly Charges (Bubble = Tickets)",
                        color_continuous_scale="Reds"
                    )
                    st.plotly_chart(apply_dark_theme(fig_scatter), use_container_width=True)

        with tab2:
            st.subheader("Global Feature Drivers")
            st.markdown("Quantifies the strongest relative indicators driving customer churn across the dataset:")
            
            with st.container(border=True):
                importance_df = pd.DataFrame({
                    'Feature': X_encoded.columns,
                    'Importance': model.feature_importances_
                }).sort_values(by='Importance', ascending=True)
                
                fig_imp = px.bar(
                    importance_df, x='Importance', y='Feature', orientation='h',
                    title="Random Forest Feature Importance Weights",
                    color='Importance', color_continuous_scale='Blues'
                )
                st.plotly_chart(apply_dark_theme(fig_imp), use_container_width=True)

        with tab3:
            st.subheader("High-Risk Retention Targets")
            
            with st.container(border=True):
                risk_threshold = st.slider("Select Churn Risk Threshold for Action List:", 0.50, 0.95, 0.70)
                
                high_risk_df = df[df['Churn_Probability'] >= risk_threshold].sort_values(
                    by='Churn_Probability', ascending=False
                ).copy()
                
                high_risk_df['Risk_Score_%'] = (high_risk_df['Churn_Probability'] * 100).round(1)
                
                st.write(f"Displaying **{len(high_risk_df)}** target customers with risk score ≥ **{int(risk_threshold*100)}%**:")
                
                st.dataframe(
                    high_risk_df[['CustomerID', 'Tenure_Months', 'Monthly_Charges', 'Contract_Type', 'Support_Tickets_Last_6M', 'Risk_Score_%']],
                    use_container_width=True,
                    column_config={
                        "Risk_Score_%": st.column_config.ProgressColumn(
                            "Churn Risk",
                            format="%.1f%%",
                            min_value=0,
                            max_value=100,
                        ),
                    }
                )
                
                csv_export = high_risk_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Export Retention Target List (CSV)",
                    data=csv_export,
                    file_name="high_risk_customers.csv",
                    mime="text/csv"
                )

        with tab4:
            st.subheader("Retention Campaign ROI Sensitivity Analysis")
            st.markdown("Estimate financial outcomes by running targeted retention campaigns.")
            
            with st.container(border=True):
                r_col1, r_col2 = st.columns(2)
                
                with r_col1:
                    retention_cost = st.number_input(
                        "Retention Offer Cost per Targeted Customer ($):",
                        min_value=1.0, max_value=500.0, value=50.0, step=5.0
                    )
                    
                with r_col2:
                    success_rate = st.slider(
                        "Expected Campaign Success Rate (% of targeted saved):",
                        min_value=5, max_value=100, value=25, step=5
                    ) / 100.0

                target_customers_count = len(predicted_churners)
                total_campaign_cost = target_customers_count * retention_cost
                gross_arr_saved = (predicted_churners['Monthly_Charges'] * 12).sum() * success_rate
                net_revenue_saved = gross_arr_saved - total_campaign_cost
                roi_pct = (net_revenue_saved / total_campaign_cost * 100) if total_campaign_cost > 0 else 0

                st.markdown("---")
                st.markdown("#### 💵 Estimated Campaign Metrics")
                f1, f2, f3, f4 = st.columns(4)
                
                f1.metric("Targeted Customers", f"{target_customers_count:,}")
                f2.metric("Total Campaign Cost", f"${total_campaign_cost:,.2f}")
                f3.metric("Gross ARR Saved", f"${gross_arr_saved:,.2f}")
                f4.metric("Net Projected ROI", f"{roi_pct:,.1f}%", delta=f"${net_revenue_saved:,.2f} Net Profit")

                financial_summary = pd.DataFrame({
                    'Metric': ['Campaign Investment', 'Gross Revenue Retained', 'Net Financial Gain'],
                    'Amount ($)': [total_campaign_cost, gross_arr_saved, net_revenue_saved]
                })
                
                fig_roi = px.bar(
                    financial_summary, x='Metric', y='Amount ($)', color='Metric',
                    title="Campaign Cost vs. Retained Revenue Breakdown",
                    text_auto='.2s', color_discrete_sequence=['#DA3633', '#238636', '#2F81F7']
                )
                st.plotly_chart(apply_dark_theme(fig_roi), use_container_width=True)

        with tab5:
            st.subheader("Technical Model Evaluation & Validation")
            st.markdown("Performance metrics evaluated on test validation data:")
            
            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, zero_division=0)
            rec = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Accuracy", f"{acc*100:.1f}%")
            m2.metric("Precision", f"{prec*100:.1f}%")
            m3.metric("Recall", f"{rec*100:.1f}%")
            m4.metric("F1-Score", f"{f1*100:.1f}%")
            
            st.markdown("<br>", unsafe_allow_html=True)
            e_col1, e_col2 = st.columns(2)
            
            with e_col1:
                with st.container(border=True):
                    st.markdown("#### Confusion Matrix")
                    cm = confusion_matrix(y_test, y_pred)
                    x_labels = ['Retained (0)', 'Churned (1)']
                    y_labels = ['Retained (0)', 'Churned (1)']
                    
                    fig_cm = px.imshow(
                        cm, 
                        x=x_labels, 
                        y=y_labels, 
                        text_auto=True, 
                        color_continuous_scale='Blues',
                        aspect="auto"
                    )
                    fig_cm.update_layout(xaxis_title="Predicted Label", yaxis_title="Actual Label")
                    fig_cm.update_coloraxes(showscale=False)
                    st.plotly_chart(apply_dark_theme(fig_cm), use_container_width=True)
                
            with e_col2:
                with st.container(border=True):
                    st.markdown("#### Receiver Operating Characteristic (ROC)")
                    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
                    roc_auc = auc(fpr, tpr)
                    
                    fig_roc = px.area(
                        x=fpr, y=tpr,
                        title=f'ROC Curve (AUC = {roc_auc:.3f})',
                        labels=dict(x='False Positive Rate', y='True Positive Rate'),
                        color_discrete_sequence=['#2F81F7']
                    )
                    fig_roc.add_shape(
                        type='line', line=dict(dash='dash', color='#8B949E'),
                        x0=0, x1=1, y0=0, y1=1
                    )
                    st.plotly_chart(apply_dark_theme(fig_roc), use_container_width=True)