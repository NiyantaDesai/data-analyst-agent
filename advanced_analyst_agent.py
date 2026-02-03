"""
Advanced Jr. Data Analyst Agent - Streamlit Application
=======================================================

A comprehensive AI-powered data analysis tool that performs:
- Automated EDA with visualizations
- Feature importance analysis
- Outlier detection and handling
- Data cleaning and transformation
- Interactive AI chatbot for insights

Installation:
pip install streamlit pandas numpy matplotlib seaborn scikit-learn anthropic plotly

Usage:
streamlit run data_analyst_agent.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from scipy import stats
import anthropic
import json
import io
import base64
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="Jr. Data Analyst Agent",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem 0;
    }
    .section-header {
        font-size: 1.8rem;
        font-weight: bold;
        color: #2ca02c;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .insight-box {
        background-color: #e8f4f8;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #17a2b8;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #ffc107;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'df' not in st.session_state:
    st.session_state.df = None
if 'cleaned_df' not in st.session_state:
    st.session_state.cleaned_df = None
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = {}
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'API_KEY' not in st.session_state:
    st.session_state.api_key = None


class DataAnalystAgent:
    """
    Advanced AI-powered data analyst that performs comprehensive
    exploratory data analysis, feature engineering, and insights generation.
    """
    
    def __init__(self, api_key=None):
        self.api_key = api_key
        if api_key:
            self.client = anthropic.Anthropic(api_key=api_key)
            self.model = "claude-sonnet-4-20250514"
        
    def analyze_data_types(self, df):
        """Categorize columns by data type and characteristics"""
        analysis = {
            'numerical': [],
            'categorical': [],
            'datetime': [],
            'binary': [],
            'high_cardinality': []
        }
        
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                unique_ratio = df[col].nunique() / len(df)
                if df[col].nunique() == 2:
                    analysis['binary'].append(col)
                else:
                    analysis['numerical'].append(col)
            elif pd.api.types.is_datetime64_any_dtype(df[col]):
                analysis['datetime'].append(col)
            else:
                unique_count = df[col].nunique()
                if unique_count > 50:
                    analysis['high_cardinality'].append(col)
                else:
                    analysis['categorical'].append(col)
        
        return analysis
    
    def detect_outliers(self, df, columns, method='iqr'):
        """Detect outliers using IQR or Z-score method"""
        outliers_info = {}
        
        for col in columns:
            if method == 'iqr':
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
            else:  # z-score
                z_scores = np.abs(stats.zscore(df[col].dropna()))
                outliers = df[z_scores > 3]
            
            outliers_info[col] = {
                'count': len(outliers),
                'percentage': (len(outliers) / len(df)) * 100,
                'indices': outliers.index.tolist()
            }
        
        return outliers_info
    
    def calculate_feature_importance(self, df, target_col, feature_cols):
        """Calculate feature importance for target prediction"""
        importance_dict = {}
        
        # Prepare data
        X = df[feature_cols].copy()
        y = df[target_col].copy()
        
        # Handle categorical variables
        le = LabelEncoder()
        for col in X.columns:
            if X[col].dtype == 'object':
                X[col] = le.fit_transform(X[col].astype(str))
        
        # Handle missing values
        X = X.fillna(X.mean())
        
        # Determine if classification or regression
        is_classification = y.nunique() < 20 or y.dtype == 'object'
        
        if is_classification:
            if y.dtype == 'object':
                y = le.fit_transform(y.astype(str))
            # Mutual information for classification
            mi_scores = mutual_info_classif(X, y, random_state=42)
            # Random Forest for classification
            rf = RandomForestClassifier(n_estimators=100, random_state=42)
            rf.fit(X, y)
            rf_importance = rf.feature_importances_
        else:
            # Mutual information for regression
            mi_scores = mutual_info_regression(X, y, random_state=42)
            # Random Forest for regression
            rf = RandomForestRegressor(n_estimators=100, random_state=42)
            rf.fit(X, y)
            rf_importance = rf.feature_importances_
        
        # Combine scores
        for i, col in enumerate(feature_cols):
            importance_dict[col] = {
                'mutual_info': mi_scores[i],
                'random_forest': rf_importance[i],
                'combined_score': (mi_scores[i] + rf_importance[i]) / 2
            }
        
        return importance_dict
    
    def check_data_quality(self, df):
        """Comprehensive data quality assessment"""
        quality_report = {
            'missing_values': {},
            'duplicates': len(df[df.duplicated()]),
            'data_types': {},
            'unique_counts': {},
            'recommendations': []
        }
        
        for col in df.columns:
            missing_count = df[col].isnull().sum()
            missing_pct = (missing_count / len(df)) * 100
            
            quality_report['missing_values'][col] = {
                'count': int(missing_count),
                'percentage': float(missing_pct)
            }
            quality_report['data_types'][col] = str(df[col].dtype)
            quality_report['unique_counts'][col] = int(df[col].nunique())
            
            # Generate recommendations
            if missing_pct > 50:
                quality_report['recommendations'].append(
                    f"⚠️ Consider dropping '{col}' - {missing_pct:.1f}% missing values"
                )
            elif missing_pct > 5:
                quality_report['recommendations'].append(
                    f"📝 '{col}' has {missing_pct:.1f}% missing - consider imputation"
                )
        
        if quality_report['duplicates'] > 0:
            quality_report['recommendations'].append(
                f"🔄 Found {quality_report['duplicates']} duplicate rows - consider removing"
            )
        
        return quality_report
    
    def clean_data(self, df, operations):
        """Apply data cleaning operations"""
        cleaned_df = df.copy()
        applied_operations = []
        
        # Remove duplicates
        if operations.get('remove_duplicates', False):
            before = len(cleaned_df)
            cleaned_df = cleaned_df.drop_duplicates()
            applied_operations.append(f"Removed {before - len(cleaned_df)} duplicate rows")
        
        # Handle missing values
        if 'missing_strategy' in operations:
            for col, strategy in operations['missing_strategy'].items():
                if strategy == 'drop':
                    cleaned_df = cleaned_df.dropna(subset=[col])
                    applied_operations.append(f"Dropped rows with missing '{col}'")
                elif strategy == 'mean':
                    cleaned_df[col].fillna(cleaned_df[col].mean(), inplace=True)
                    applied_operations.append(f"Filled '{col}' missing with mean")
                elif strategy == 'median':
                    cleaned_df[col].fillna(cleaned_df[col].median(), inplace=True)
                    applied_operations.append(f"Filled '{col}' missing with median")
                elif strategy == 'mode':
                    cleaned_df[col].fillna(cleaned_df[col].mode()[0], inplace=True)
                    applied_operations.append(f"Filled '{col}' missing with mode")
        
        # Handle outliers
        if 'outlier_strategy' in operations:
            for col, strategy in operations['outlier_strategy'].items():
                if strategy == 'remove':
                    Q1 = cleaned_df[col].quantile(0.25)
                    Q3 = cleaned_df[col].quantile(0.75)
                    IQR = Q3 - Q1
                    before = len(cleaned_df)
                    cleaned_df = cleaned_df[
                        (cleaned_df[col] >= Q1 - 1.5 * IQR) & 
                        (cleaned_df[col] <= Q3 + 1.5 * IQR)
                    ]
                    applied_operations.append(f"Removed {before - len(cleaned_df)} outliers from '{col}'")
                elif strategy == 'cap':
                    Q1 = cleaned_df[col].quantile(0.25)
                    Q3 = cleaned_df[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower = Q1 - 1.5 * IQR
                    upper = Q3 + 1.5 * IQR
                    cleaned_df[col] = cleaned_df[col].clip(lower, upper)
                    applied_operations.append(f"Capped outliers in '{col}'")
        
        return cleaned_df, applied_operations
    
    def get_ai_insights(self, context, question=None):
        """Get AI-powered insights using Claude"""
        if not self.api_key:
            return "Please provide an API key to use AI insights."
        
        prompt = f"""You are an expert data analyst. Based on the following data analysis context:

{json.dumps(context, indent=2)}

{"Answer this specific question: " + question if question else "Provide key insights, patterns, and recommendations for further analysis. Focus on actionable insights and potential areas of concern or opportunity."}

Be specific, technical where appropriate, and provide clear recommendations."""
        
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=2000,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.content[0].text
        except Exception as e:
            return f"Error getting AI insights: {str(e)}"


def create_distribution_plots(df, numerical_cols):
    """Create distribution plots for numerical columns"""
    plots = []
    
    for col in numerical_cols[:6]:  # Limit to 6 plots
        fig = go.Figure()
        
        # Histogram
        fig.add_trace(go.Histogram(
            x=df[col].dropna(),
            name='Distribution',
            nbinsx=30,
            marker_color='steelblue',
            opacity=0.7
        ))
        
        fig.update_layout(
            title=f'Distribution of {col}',
            xaxis_title=col,
            yaxis_title='Frequency',
            showlegend=True,
            height=400
        )
        
        plots.append((col, fig))
    
    return plots


def create_correlation_heatmap(df, numerical_cols):
    """Create correlation heatmap"""
    corr_matrix = df[numerical_cols].corr()
    
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        colorscale='RdBu',
        zmid=0,
        text=corr_matrix.values.round(2),
        texttemplate='%{text}',
        textfont={"size": 10},
        colorbar=dict(title="Correlation")
    ))
    
    fig.update_layout(
        title='Correlation Matrix',
        width=800,
        height=800
    )
    
    return fig


def create_boxplots(df, numerical_cols):
    """Create boxplots for outlier visualization"""
    plots = []
    
    for col in numerical_cols[:6]:
        fig = go.Figure()
        
        fig.add_trace(go.Box(
            y=df[col].dropna(),
            name=col,
            marker_color='lightseagreen',
            boxmean='sd'
        ))
        
        fig.update_layout(
            title=f'Boxplot of {col}',
            yaxis_title=col,
            showlegend=False,
            height=400
        )
        
        plots.append((col, fig))
    
    return plots


def create_feature_importance_chart(importance_dict):
    """Create feature importance visualization"""
    # Sort by combined score
    sorted_features = sorted(
        importance_dict.items(),
        key=lambda x: x[1]['combined_score'],
        reverse=True
    )
    
    features = [f[0] for f in sorted_features]
    mi_scores = [f[1]['mutual_info'] for f in sorted_features]
    rf_scores = [f[1]['random_forest'] for f in sorted_features]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        y=features,
        x=mi_scores,
        name='Mutual Information',
        orientation='h',
        marker_color='steelblue'
    ))
    
    fig.add_trace(go.Bar(
        y=features,
        x=rf_scores,
        name='Random Forest',
        orientation='h',
        marker_color='lightcoral'
    ))
    
    fig.update_layout(
        title='Feature Importance Analysis',
        xaxis_title='Importance Score',
        yaxis_title='Features',
        barmode='group',
        height=max(400, len(features) * 30)
    )
    
    return fig


# Main App
def main():
    st.markdown('<div class="main-header">🤖 Advanced Jr. Data Analyst Agent</div>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Sidebar
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # API Key input - Try to get from secrets first, then allow user input
        try:
            # Try to get API key from Streamlit secrets
            default_api_key = st.secrets.get("ANTHROPIC_API_KEY", None)
            if default_api_key:
                st.success("✅ API Key loaded from secure storage")
                api_key = default_api_key
                # Option to override with custom key
                use_custom = st.checkbox("Use different API key", value=False)
                if use_custom:
                    api_key = st.text_input(
                        "Custom Anthropic API Key",
                        type="password",
                        help="Enter a different API key"
                    )
            else:
                # No secret found, ask user for key
                api_key = st.text_input(
                    "Anthropic API Key",
                    type="password",
                    help="Enter your Anthropic API key for AI-powered insights"
                )
        except Exception as e:
            # Fallback if secrets not available (local development)
            api_key = st.text_input(
                "Anthropic API Key",
                type="password",
                help="Enter your Anthropic API key for AI-powered insights"
            )
        
        if api_key:
            st.session_state.api_key = API_KEY
        
        st.markdown("---")
        
        # File upload
        st.header("📂 Data Upload")
        uploaded_file = st.file_uploader(
            "Upload CSV file",
            type=['csv'],
            help="Upload your dataset for analysis"
        )
        
        if uploaded_file:
            try:
                df = pd.read_csv(uploaded_file)
                st.session_state.df = df
                st.success(f"✅ Loaded {len(df)} rows, {len(df.columns)} columns")
            except Exception as e:
                st.error(f"Error loading file: {str(e)}")
        
        st.markdown("---")
        
        # Sample data option
        if st.button("📊 Load Sample Data"):
            np.random.seed(42)
            n = 500
            
            sample_df = pd.DataFrame({
                'customer_id': range(1, n+1),
                'age': np.random.randint(18, 70, n),
                'income': np.random.normal(50000, 15000, n),
                'credit_score': np.random.randint(300, 850, n),
                'loan_amount': np.random.normal(20000, 8000, n),
                'employment_years': np.random.randint(0, 40, n),
                'num_credit_cards': np.random.poisson(3, n),
                'debt_ratio': np.random.uniform(0, 1, n),
                'region': np.random.choice(['North', 'South', 'East', 'West'], n),
                'education': np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], n),
                'default': np.random.choice([0, 1], n, p=[0.85, 0.15])
            })
            
            # Add some missing values
            sample_df.loc[np.random.choice(sample_df.index, 20), 'income'] = np.nan
            sample_df.loc[np.random.choice(sample_df.index, 15), 'credit_score'] = np.nan
            
            st.session_state.df = sample_df
            st.success("✅ Sample dataset loaded!")
            st.rerun()
    
    # Main content
    if st.session_state.df is not None:
        df = st.session_state.df
        agent = DataAnalystAgent(st.session_state.api_key)
        
        # Tabs for different sections
        tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
            "📊 Overview",
            "📈 Distributions",
            "🔗 Correlations",
            "🎯 Feature Importance",
            "⚠️ Outliers",
            "🧹 Data Cleaning",
            "💬 AI Assistant"
        ])
        
        # Tab 1: Overview
        with tab1:
            st.markdown('<div class="section-header">Dataset Overview</div>', unsafe_allow_html=True)
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Rows", f"{len(df):,}")
            with col2:
                st.metric("Total Columns", len(df.columns))
            with col3:
                st.metric("Missing Values", f"{df.isnull().sum().sum():,}")
            with col4:
                st.metric("Duplicates", f"{len(df[df.duplicated()]):,}")
            
            st.markdown("### Data Preview")
            st.dataframe(df.head(20), use_container_width=True)
            
            # Data types analysis
            st.markdown("### Column Analysis")
            type_analysis = agent.analyze_data_types(df)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Numerical Columns:**")
                for col in type_analysis['numerical']:
                    st.write(f"- {col}")
                
                st.markdown("**Binary Columns:**")
                for col in type_analysis['binary']:
                    st.write(f"- {col}")
            
            with col2:
                st.markdown("**Categorical Columns:**")
                for col in type_analysis['categorical']:
                    st.write(f"- {col}")
                
                st.markdown("**High Cardinality:**")
                for col in type_analysis['high_cardinality']:
                    st.write(f"- {col}")
            
            # Statistical summary
            st.markdown("### Statistical Summary")
            st.dataframe(df.describe(), use_container_width=True)
            
            # Data quality report
            st.markdown("### Data Quality Report")
            quality_report = agent.check_data_quality(df)
            
            if quality_report['recommendations']:
                st.markdown('<div class="warning-box">', unsafe_allow_html=True)
                st.markdown("**Recommendations:**")
                for rec in quality_report['recommendations']:
                    st.markdown(f"- {rec}")
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Missing values visualization
            if df.isnull().sum().sum() > 0:
                missing_df = pd.DataFrame({
                    'Column': df.columns,
                    'Missing Count': df.isnull().sum().values,
                    'Missing %': (df.isnull().sum().values / len(df) * 100).round(2)
                })
                missing_df = missing_df[missing_df['Missing Count'] > 0].sort_values('Missing Count', ascending=False)
                
                fig = px.bar(
                    missing_df,
                    x='Column',
                    y='Missing %',
                    title='Missing Values by Column',
                    color='Missing %',
                    color_continuous_scale='Reds'
                )
                st.plotly_chart(fig, use_container_width=True)
        
        # Tab 2: Distributions
        with tab2:
            st.markdown('<div class="section-header">Distribution Analysis</div>', unsafe_allow_html=True)
            
            type_analysis = agent.analyze_data_types(df)
            numerical_cols = type_analysis['numerical'] + type_analysis['binary']
            
            if numerical_cols:
                dist_plots = create_distribution_plots(df, numerical_cols)
                
                for col, fig in dist_plots:
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Statistical tests
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Mean", f"{df[col].mean():.2f}")
                    with col2:
                        st.metric("Median", f"{df[col].median():.2f}")
                    with col3:
                        st.metric("Std Dev", f"{df[col].std():.2f}")
                    
                    # Normality test
                    if len(df[col].dropna()) > 3:
                        stat, p_value = stats.normaltest(df[col].dropna())
                        if p_value < 0.05:
                            st.info(f"📊 Distribution appears non-normal (p={p_value:.4f})")
                        else:
                            st.success(f"✅ Distribution appears normal (p={p_value:.4f})")
                    
                    st.markdown("---")
            else:
                st.warning("No numerical columns found for distribution analysis")
        
        # Tab 3: Correlations
        with tab3:
            st.markdown('<div class="section-header">Correlation Analysis</div>', unsafe_allow_html=True)
            
            if numerical_cols:
                fig = create_correlation_heatmap(df, numerical_cols)
                st.plotly_chart(fig, use_container_width=True)
                
                # High correlations
                corr_matrix = df[numerical_cols].corr()
                high_corr = []
                
                for i in range(len(corr_matrix.columns)):
                    for j in range(i+1, len(corr_matrix.columns)):
                        if abs(corr_matrix.iloc[i, j]) > 0.7:
                            high_corr.append({
                                'Feature 1': corr_matrix.columns[i],
                                'Feature 2': corr_matrix.columns[j],
                                'Correlation': corr_matrix.iloc[i, j]
                            })
                
                if high_corr:
                    st.markdown("### High Correlations (|r| > 0.7)")
                    st.dataframe(pd.DataFrame(high_corr), use_container_width=True)
                else:
                    st.info("No high correlations found")
            else:
                st.warning("No numerical columns for correlation analysis")
        
        # Tab 4: Feature Importance
        with tab4:
            st.markdown('<div class="section-header">Feature Importance Analysis</div>', unsafe_allow_html=True)
            
            # Select target variable
            target_col = st.selectbox(
                "Select Target Variable",
                df.columns.tolist(),
                help="Choose the column you want to predict"
            )
            
            if target_col:
                feature_cols = [col for col in df.columns if col != target_col]
                
                if st.button("Calculate Feature Importance"):
                    with st.spinner("Calculating feature importance..."):
                        try:
                            importance_dict = agent.calculate_feature_importance(
                                df, target_col, feature_cols
                            )
                            
                            st.session_state.analysis_results['feature_importance'] = importance_dict
                            
                            # Visualize
                            fig = create_feature_importance_chart(importance_dict)
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Top features
                            sorted_features = sorted(
                                importance_dict.items(),
                                key=lambda x: x[1]['combined_score'],
                                reverse=True
                            )
                            
                            st.markdown("### Top 10 Most Important Features")
                            top_features_df = pd.DataFrame([
                                {
                                    'Feature': f[0],
                                    'Combined Score': f"{f[1]['combined_score']:.4f}",
                                    'Mutual Info': f"{f[1]['mutual_info']:.4f}",
                                    'Random Forest': f"{f[1]['random_forest']:.4f}"
                                }
                                for f in sorted_features[:10]
                            ])
                            st.dataframe(top_features_df, use_container_width=True)
                            
                        except Exception as e:
                            st.error(f"Error calculating feature importance: {str(e)}")
        
        # Tab 5: Outliers
        with tab5:
            st.markdown('<div class="section-header">Outlier Detection</div>', unsafe_allow_html=True)
            
            if numerical_cols:
                outlier_method = st.selectbox(
                    "Detection Method",
                    ['IQR (Interquartile Range)', 'Z-Score'],
                    help="Choose method for outlier detection"
                )
                
                method = 'iqr' if 'IQR' in outlier_method else 'zscore'
                
                outliers_info = agent.detect_outliers(df, numerical_cols, method)
                
                # Summary
                st.markdown("### Outlier Summary")
                outlier_summary = pd.DataFrame([
                    {
                        'Column': col,
                        'Outliers': info['count'],
                        'Percentage': f"{info['percentage']:.2f}%"
                    }
                    for col, info in outliers_info.items()
                    if info['count'] > 0
                ])
                
                if len(outlier_summary) > 0:
                    st.dataframe(outlier_summary, use_container_width=True)
                else:
                    st.success("✅ No outliers detected!")
                
                # Boxplots
                st.markdown("### Outlier Visualization")
                boxplots = create_boxplots(df, numerical_cols)
                
                for col, fig in boxplots:
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No numerical columns for outlier detection")
        
        # Tab 6: Data Cleaning
        with tab6:
            st.markdown('<div class="section-header">Data Cleaning & Transformation</div>', unsafe_allow_html=True)
            
            st.markdown("### Cleaning Operations")
            
            operations = {}
            
            # Duplicates
            if len(df[df.duplicated()]) > 0:
                operations['remove_duplicates'] = st.checkbox(
                    f"Remove {len(df[df.duplicated()])} duplicate rows"
                )
            
            # Missing values
            st.markdown("### Handle Missing Values")
            missing_strategy = {}
            
            for col in df.columns:
                missing_count = df[col].isnull().sum()
                if missing_count > 0:
                    strategy = st.selectbox(
                        f"{col} ({missing_count} missing)",
                        ['Keep', 'Drop Rows', 'Mean', 'Median', 'Mode'],
                        key=f"missing_{col}"
                    )
                    
                    if strategy != 'Keep':
                        missing_strategy[col] = strategy.lower().replace(' rows', '')
            
            if missing_strategy:
                operations['missing_strategy'] = missing_strategy
            
            # Outliers
            st.markdown("### Handle Outliers")
            outlier_strategy = {}
            
            outliers_info = agent.detect_outliers(df, numerical_cols, 'iqr')
            
            for col, info in outliers_info.items():
                if info['count'] > 0:
                    strategy = st.selectbox(
                        f"{col} ({info['count']} outliers)",
                        ['Keep', 'Remove', 'Cap at IQR bounds'],
                        key=f"outlier_{col}"
                    )
                    
                    if strategy != 'Keep':
                        outlier_strategy[col] = strategy.lower().split()[0]
            
            if outlier_strategy:
                operations['outlier_strategy'] = outlier_strategy
            
            # Apply cleaning
            if st.button("🧹 Apply Cleaning Operations", type="primary"):
                if operations:
                    with st.spinner("Cleaning data..."):
                        cleaned_df, applied_ops = agent.clean_data(df, operations)
                        st.session_state.cleaned_df = cleaned_df
                        
                        st.success(f"✅ Data cleaned! {len(df) - len(cleaned_df)} rows removed")
                        
                        st.markdown("### Operations Applied:")
                        for op in applied_ops:
                            st.write(f"- {op}")
                        
                        # Comparison
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Original Rows", len(df))
                        with col2:
                            st.metric("Cleaned Rows", len(cleaned_df))
                        
                        # Download cleaned data
                        csv = cleaned_df.to_csv(index=False)
                        st.download_button(
                            "📥 Download Cleaned Data",
                            csv,
                            "cleaned_data.csv",
                            "text/csv"
                        )
                else:
                    st.warning("No cleaning operations selected")
        
        # Tab 7: AI Assistant
        with tab7:
            st.markdown('<div class="section-header">AI Data Analysis Assistant</div>', unsafe_allow_html=True)
            
            if not st.session_state.api_key:
                st.warning("⚠️ Please enter your Anthropic API key in the sidebar to use the AI Assistant")
            else:
                st.markdown("Ask questions about your data, request insights, or get help understanding the analysis!")
                
                # Prepare context for AI
                context = {
                    'dataset_shape': {'rows': len(df), 'columns': len(df.columns)},
                    'columns': df.columns.tolist(),
                    'data_types': agent.analyze_data_types(df),
                    'missing_values': df.isnull().sum().to_dict(),
                    'statistical_summary': df.describe().to_dict(),
                    'quality_report': agent.check_data_quality(df)
                }
                
                if 'feature_importance' in st.session_state.analysis_results:
                    context['feature_importance'] = st.session_state.analysis_results['feature_importance']
                
                # Chat interface
                st.markdown("### Chat History")
                
                # Display chat history
                for msg in st.session_state.chat_history:
                    with st.chat_message(msg['role']):
                        st.markdown(msg['content'])
                
                # Chat input
                user_question = st.chat_input("Ask me anything about your data...")
                
                if user_question:
                    # Add user message
                    st.session_state.chat_history.append({
                        'role': 'user',
                        'content': user_question
                    })
                    
                    with st.chat_message("user"):
                        st.markdown(user_question)
                    
                    # Get AI response
                    with st.chat_message("assistant"):
                        with st.spinner("Analyzing..."):
                            response = agent.get_ai_insights(context, user_question)
                            st.markdown(response)
                            
                            st.session_state.chat_history.append({
                                'role': 'assistant',
                                'content': response
                            })
                
                # Quick insights button
                st.markdown("---")
                if st.button("💡 Generate Quick Insights"):
                    with st.spinner("Generating insights..."):
                        insights = agent.get_ai_insights(context)
                        
                        st.markdown('<div class="insight-box">', unsafe_allow_html=True)
                        st.markdown("### AI-Generated Insights")
                        st.markdown(insights)
                        st.markdown('</div>', unsafe_allow_html=True)
                        
                        st.session_state.chat_history.append({
                            'role': 'assistant',
                            'content': insights
                        })
                
                # Suggested questions
                st.markdown("### Suggested Questions")
                col1, col2 = st.columns(2)
                
                with col1:
                    if st.button("📊 What are the main patterns in this data?"):
                        st.rerun()
                    if st.button("🎯 Which features are most important?"):
                        st.rerun()
                
                with col2:
                    if st.button("⚠️ What data quality issues should I address?"):
                        st.rerun()
                    if st.button("📈 What visualizations should I create next?"):
                        st.rerun()
    
    else:
        # Welcome screen
        st.markdown("""
        <div style='text-align: center; padding: 3rem;'>
            <h2>Welcome to the Advanced Jr. Data Analyst Agent! 🚀</h2>
            <p style='font-size: 1.2rem; color: #666;'>
                Upload your dataset or load sample data to get started with comprehensive AI-powered analysis.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            st.markdown("""
            ### What This Tool Does:
            
            ✅ **Automated EDA** - Comprehensive exploratory data analysis
            
            📊 **Smart Visualizations** - Distribution, correlation, and outlier plots
            
            🎯 **Feature Importance** - Identify which variables matter most
            
            ⚠️ **Outlier Detection** - Find and handle anomalous data points
            
            🧹 **Data Cleaning** - Remove duplicates, handle missing values
            
            🤖 **AI Assistant** - Chat with an AI to understand your data better
            
            💡 **Actionable Insights** - Get recommendations for next steps
            """)
            
            st.markdown("---")
            st.info("👈 Upload your data in the sidebar to begin!")


if __name__ == "__main__":
    main()
            
