import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib
matplotlib.use('Agg')  # Set non-interactive backend before importing pyplot
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from lime import lime_tabular
import plotly.express as px
import plotly.graph_objects as go
from sklearn.inspection import permutation_importance, partial_dependence

# Set page configuration
st.set_page_config(
    page_title="Financial Fraud Detection with XAI",
    page_icon="💰",
    layout="wide"
)  


# Load the trained model
@st.cache_resource
def load_model():
    return pickle.load(open('final_model1.sav', 'rb'))

loaded_model = load_model()

# Load sample data for explanations
@st.cache_data
def load_sample_data():
    # Replace with actual path if different
    try:
        sample_data = pd.read_csv('sample_data.csv')
    except:
        # Create small synthetic dataset if sample data not available
        # Using only the 10 features that the model expects
        sample_data = pd.DataFrame({
            'card1': np.random.randint(1000, 10000, 100),
            'card2': np.random.randint(100, 500, 100),
            'card4': np.random.randint(1, 5, 100),
            'card6': np.random.randint(1, 3, 100),
            'addr1': np.random.randint(100, 500, 100),
            'addr2': np.random.randint(0, 100, 100),
            'TransactionAmt': np.random.randint(10, 20000, 100),
            'P_emaildomain': np.random.randint(0, 5, 100),
            'ProductCD': np.random.randint(0, 5, 100),
            'DeviceType': np.random.randint(1, 3, 100),
            'isFraud': np.random.choice([0, 1], 100, p=[0.97, 0.03])
        })
    return sample_data

sample_data = load_sample_data()

# Important features based on XAI analysis
important_features = ['card1', 'card2', 'card4', 'card6', 'addr1', 'addr2', 
                     'TransactionAmt', 'P_emaildomain', 'ProductCD', 'DeviceType']

# Generate SHAP values
@st.cache_data
def get_shap_values():
    # Get a subset of data for SHAP explanations
    X_subset = sample_data.drop('isFraud', axis=1, errors='ignore').iloc[:100]
    
    try:
        # Initialize SHAP explainer
        explainer = shap.TreeExplainer(loaded_model)
        
        # Calculate SHAP values for the subset
        shap_values = explainer.shap_values(X_subset)
        
        # For binary classification with LightGBM, shap_values is a list of two ndarrays
        # First element is negative class, second is positive class
        if isinstance(shap_values, list) and len(shap_values) == 2:
            # For binary classification, return the explainer, shap_values list, and data
            return explainer, shap_values, X_subset
        else:
            # For other cases, wrap in a list to maintain consistent format
            return explainer, [shap_values], X_subset
    except Exception as e:
        st.error(f"Error calculating SHAP values: {e}")
        # Return dummy values for graceful failure
        dummy_values = np.zeros((X_subset.shape[0], X_subset.shape[1]))
        return None, [dummy_values], X_subset

# Helper function for creating Plotly-based SHAP force plots to avoid Matplotlib backend issues
def create_force_plot_plotly(shap_values, features, feature_names):
    """
    Create a SHAP force plot using Plotly instead of Matplotlib
    """
    # Get the base value and the contribution of each feature
    data = pd.DataFrame({
        'Feature': feature_names,
        'Value': features,
        'Contribution': shap_values
    })
    
    # Sort by absolute contribution
    data['AbsContribution'] = np.abs(data['Contribution'])
    data = data.sort_values('AbsContribution', ascending=False)
    
    # Create color scheme
    colors = ['#ff4d4d' if x < 0 else '#4d94ff' for x in data['Contribution']]
    
    # Create the bar chart
    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=data['Feature'],
        x=data['Contribution'],
        orientation='h',
        marker_color=colors,
        text=[f"{name}: {value}" for name, value in zip(data['Feature'], data['Value'])],
        hoverinfo='text+x'
    ))
    
    fig.update_layout(
        title='SHAP Force Plot',
        xaxis_title='Feature Contribution',
        height=400,
        margin=dict(l=10, r=10, t=40, b=10)
    )
    
    return fig

# Main application layout
st.sidebar.title("Navigation")
app_mode = st.sidebar.radio("Select Mode", ["Prediction", "XAI Dashboard"])

if app_mode == "Prediction":
    st.title("Financial Transaction Fraud Prediction")
    st.write("Enter transaction details to predict if it's fraudulent or legitimate.")
    
    # Add feature descriptions in a collapsible section
    with st.expander("Feature Descriptions Reference"):
        st.markdown("""
        | Feature | Description |
        | ------- | ----------- |
        | Card 1 | Unique card identifier or first digits of card number (anonymized) |
        | Card 2 | Card issuer identifier or bin number (range 100-600) |
        | Card 4 | Card network/category (Discover, Mastercard, AmEx, Visa) |
        | Card 6 | Type of card (Credit or Debit) |
        | Address 1 | Numeric representation of billing zip code |
        | Address 2 | Numeric code representing billing country |
        | Transaction Amount | The transaction amount in USD |
        | P Email Domain | Purchaser's email domain category |
        | Product Code | Product category for the transaction |
        | Device Type | Device used for the transaction (Mobile or Desktop) |
        """)
    
    # Create columns for input fields
    col1, col2 = st.columns(2)
    with col1:
        card1 = st.number_input("Card 1", min_value=1000, max_value=20000, value=5000, 
                              help="Unique card identifier or first digits of card number (anonymized)")
        card2 = st.number_input("Card 2", min_value=100, max_value=600, value=300,
                              help="Card issuer identifier or bin number (range 100-600)")
        card4 = st.selectbox("Card 4 (Category)", [1, 2, 3, 4], 
                           format_func=lambda x: {1: "Discover", 2: "Mastercard", 3: "American Express", 4: "Visa"}[x],
                           help="Card network or category")
        card6 = st.selectbox("Card 6 (Type)", [1, 2],
                           format_func=lambda x: {1: "Credit", 2: "Debit"}[x],
                           help="Type of card used for transaction")
        addr1 = st.slider("Address 1 (Zip Code)", 0, 500, 200,
                        help="Numeric representation of billing zip code")
        
    with col2:
        addr2 = st.slider("Address 2 (Country Code)", 0, 100, 50,
                        help="Numeric code representing billing country")
        TransactionAmt = st.number_input("Transaction Amount (USD)", 0, 20000, 1000,
                                      help="Dollar amount of the transaction")
        P_emaildomain = st.selectbox("P Email Domain", [0, 1, 2, 3, 4],
                                   format_func=lambda x: {0: "Gmail", 1: "Outlook", 2: "Mail.com", 3: "Others", 4: "Yahoo"}[x],
                                   help="Purchaser's email domain category")
        ProductCD = st.selectbox("Product Code", [0, 1, 2, 3, 4],
                               format_func=lambda x: {0: "C", 1: "H", 2: "R", 3: "S", 4: "W"}[x],
                               help="Category of product in transaction (C=physical, H=digital, etc.)")
        DeviceType = st.selectbox("Device Type", [1, 2],
                                format_func=lambda x: {1: "Mobile", 2: "Desktop"}[x],
                                help="Type of device used for the transaction")
    
    # Create a dictionary with input values
    input_data = {
        'card1': card1,
        'card2': card2,
        'card4': card4,
        'card6': card6,
        'addr1': addr1,
        'addr2': addr2,
        'TransactionAmt': TransactionAmt,
        'P_emaildomain': P_emaildomain,
        'ProductCD': ProductCD,
        'DeviceType': DeviceType
    }
    
    # Add any missing columns that the model expects
    required_columns = set(sample_data.columns) - {'isFraud'}
    for col in required_columns:
        if col not in input_data:
            input_data[col] = 0  # Default value
    
    # Create a DataFrame for prediction
    input_df = pd.DataFrame([input_data])
    
    if st.button("Predict"):
        try:
            # Make prediction
            prediction = loaded_model.predict_proba(input_df)[0][1]
            
            # Display prediction result
            st.subheader("Prediction Result")
            col1, col2 = st.columns(2)
            with col1:
                st.metric(label="Fraud Probability", value=f"{prediction:.2%}")
            with col2:
                if prediction > 0.5:
                    st.error("⚠️ High likelihood of fraud detected!")
                else:
                    st.success("✅ Transaction appears legitimate")
            
            # Add SHAP explanation for this prediction
            st.subheader("Explanation of Prediction")
            
            explainer, shap_values, _ = get_shap_values()
            if explainer is not None:
                input_shap_values = explainer.shap_values(input_df)
                
                # Handle different SHAP value formats for binary classification
                shap_index = 1 if (isinstance(input_shap_values, list) and len(input_shap_values) > 1) else 0
                if isinstance(input_shap_values, list):
                    shap_for_display = input_shap_values[shap_index][0]
                else:
                    shap_for_display = input_shap_values[0]
                
                # Use Plotly for force plot instead of Matplotlib
                st.write("Feature Contribution to Prediction:")
                force_plot_fig = create_force_plot_plotly(
                    shap_for_display, 
                    input_df.iloc[0].values, 
                    input_df.columns
                )
                st.plotly_chart(force_plot_fig, use_container_width=True)
                
                # Display top features influencing this prediction
                st.write("Top Features Influencing this Prediction:")
                feature_importance = pd.DataFrame({
                    'Feature': input_df.columns,
                    'Importance': np.abs(shap_for_display)
                }).sort_values('Importance', ascending=False).head(5)
                
                fig = px.bar(
                    feature_importance,
                    x='Importance',
                    y='Feature',
                    orientation='h',
                    title="Top 5 Features",
                    color='Importance',
                    color_continuous_scale=px.colors.sequential.Bluered
                )
                st.plotly_chart(fig)
            else:
                st.warning("SHAP explanation unavailable due to an error with the explainer.")
        
        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")
            st.error("Please check your input data and try again.")

else:  # XAI Dashboard mode
    st.title("XAI Dashboard for Financial Fraud Detection")
    
    # Create tabs for different XAI methods - removed "Partial Dependence Plots"
    tab1, tab2, tab3 = st.tabs([
        "Feature Importance", 
        "SHAP Analysis", 
        "LIME Explanations"
    ])
    
    with tab1:
        st.header("Feature Importance")
        st.write("Understanding which features are most important in the fraud detection model.")
        
        # Extract feature importance from the model if it's a tree-based model
        if hasattr(loaded_model, 'feature_importances_'):
            try:
                importances = loaded_model.feature_importances_
                feature_names = sample_data.drop('isFraud', axis=1, errors='ignore').columns
                
                # Check if lengths match
                if len(importances) == len(feature_names):
                    feature_importance = pd.DataFrame({
                        'Feature': feature_names,
                        'Importance': importances
                    }).sort_values('Importance', ascending=False)
                    
                    # Display importance plot
                    fig = px.bar(
                        feature_importance.head(15),  #  features
                        x='Importance',
                        y='Feature',
                        orientation='h',
                        title="Feature Importance (Top 10)",
                        color='Importance',
                        color_continuous_scale=px.colors.sequential.Viridis
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Display importance table
                    st.write("Feature Importance Table:")
                    st.dataframe(feature_importance)
                else:
                    # Handle the case when lengths don't match
                    st.warning(f"Feature count mismatch: Model has {len(importances)} features, but data has {len(feature_names)} columns.")
                    
                    # Create generic feature names
                    generic_names = [f"feature_{i}" for i in range(len(importances))]
                    feature_importance = pd.DataFrame({
                        'Feature': generic_names,
                        'Importance': importances
                    }).sort_values('Importance', ascending=False)
                    
                    # Display importance plot with generic names
                    fig = px.bar(
                        feature_importance.head(15),
                        x='Importance',
                        y='Feature',
                        orientation='h',
                        title="Feature Importance (Top 10) - Generic Feature Names",
                        color='Importance',
                        color_continuous_scale=px.colors.sequential.Viridis
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Display importance table
                    st.write("Feature Importance Table (Generic Names):")
                    st.dataframe(feature_importance)
            except Exception as e:
                st.error(f"Error displaying feature importance: {e}")
            
        else:
            st.warning("Feature importance not directly available for this model type.")
            
            # Alternative: Show permutation importance
            st.write("Calculating permutation importance instead...")
            X = sample_data.drop('isFraud', axis=1, errors='ignore')
            y = sample_data['isFraud'] if 'isFraud' in sample_data.columns else None
            
            if y is not None:
                perm_importance = permutation_importance(loaded_model, X, y, n_repeats=5, random_state=42)
                perm_importance_df = pd.DataFrame({
                    'Feature': X.columns,
                    'Importance': perm_importance.importances_mean
                }).sort_values('Importance', ascending=False)
                
                fig = px.bar(
                    perm_importance_df.head(15),
                    x='Importance',
                    y='Feature',
                    orientation='h',
                    title="Permutation Importance (Top 10)",
                    color='Importance',
                    color_continuous_scale=px.colors.sequential.Viridis
                )
                st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.header("SHAP (SHapley Additive exPlanations) Analysis")
        st.write("""
        SHAP values represent the contribution of each feature to the prediction.
        They are based on cooperative game theory and provide both global and local explanations.
        """)
        
        # Get SHAP explainer and values
        explainer, shap_values, X_subset = get_shap_values()
        
        if explainer is None:
            st.error("Could not calculate SHAP values. Using placeholder visualizations.")
        else:
            # Display SHAP summary plot
            st.subheader("SHAP Summary Plot")
            st.write("How features impact model predictions across the entire dataset")
            
            shap_summary_option = st.radio(
                "Summary plot type:",
                ["Bar", "Beeswarm", "Violin"]
            )
            
            # Get the appropriate SHAP values - for binary classification, use class 1 (positive class)
            shap_index = 1 if (isinstance(shap_values, list) and len(shap_values) > 1) else 0
            
            try:
                # Create a new figure for each plot to avoid backend issues
                fig, ax = plt.subplots(figsize=(10, 10))
                
                if shap_summary_option == "Bar":
                    shap.summary_plot(shap_values[shap_index], X_subset, plot_type="bar", show=False)
                elif shap_summary_option == "Beeswarm":
                    shap.summary_plot(shap_values[shap_index], X_subset, show=False)
                else:  # Violin
                    shap.summary_plot(shap_values[shap_index], X_subset, plot_type="violin", show=False)
                
                # Ensure tight layout
                plt.tight_layout()
                # Display the plot using Streamlit
                st.pyplot(fig)
                
            except Exception as e:
                st.error(f"Error creating SHAP summary plot: {e}")
                st.info("Creating alternative SHAP visualization...")
                
                # Alternative: Show feature importance based on mean absolute SHAP values
                shap_importance = pd.DataFrame({
                    'Feature': X_subset.columns,
                    'Importance': np.abs(shap_values[shap_index]).mean(axis=0)
                }).sort_values('Importance', ascending=False)
                
                fig = px.bar(
                    shap_importance,
                    x='Importance', 
                    y='Feature',
                    orientation='h',
                    title="Feature Importance based on SHAP Values",
                    color='Importance',
                    color_continuous_scale=px.colors.sequential.Blues
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Allow user to explore individual SHAP dependence plots
            st.subheader("SHAP Dependence Plots")
            st.write("Explore how specific features affect predictions")
            
            # Select feature for dependence plot
            feature = st.selectbox("Select feature to explore:", important_features)
            
            # Get the index of the feature in the columns list
            try:
                feature_idx = list(X_subset.columns).index(feature)
                
                # Extract the SHAP values and feature values
                shap_values_feat = shap_values[shap_index][:, feature_idx]
                feature_values = X_subset[feature].values
                
                # Create a custom dependence plot with Plotly
                fig = px.scatter(
                    x=feature_values, 
                    y=shap_values_feat,
                    title=f"SHAP Dependence Plot for {feature}",
                    labels={"x": feature, "y": "SHAP value"},
                    opacity=0.7,
                    color=np.abs(shap_values_feat),
                    color_continuous_scale="Viridis"
                )
                
                # Add a horizontal line at y=0
                fig.add_hline(y=0, line_dash="dash", line_color="gray")
                
                # Add smoothed line to show trend
                fig.add_trace(
                    go.Scatter(
                        x=feature_values[np.argsort(feature_values)],
                        y=shap_values_feat[np.argsort(feature_values)],
                        mode='lines',
                        line=dict(color='red', width=2, shape='spline', smoothing=0.6),
                        name='Trend',
                        opacity=0.7
                    )
                )
                
                fig.update_layout(height=500)
                st.plotly_chart(fig, use_container_width=True)
                
                # Add interpretation
                st.write(f"""
                **Interpretation**: This plot shows how the SHAP value (impact on model output) of {feature} 
                changes with different values of the feature. A higher SHAP value indicates a stronger push 
                towards predicting fraud.
                """)
                
            except Exception as e:
                st.error(f"Error creating dependence plot: {e}")
                fig, ax = plt.subplots(figsize=(10, 6))
                shap.dependence_plot(
                    feature, 
                    shap_values[shap_index], 
                    X_subset, 
                    ax=ax,
                    interaction_index=None,
                    show=False
                )
                plt.tight_layout()
                st.pyplot(fig)
    
    with tab3:
        st.header("LIME (Local Interpretable Model-agnostic Explanations)")
        st.write("""
        LIME explains individual predictions by learning an interpretable model 
        locally around the prediction. It helps understand why a particular 
        prediction was made for a specific instance.
        """)
        
        # Select a sample to explain
        sample_index = st.slider(
            "Select sample to explain:", 
            0, 
            len(sample_data)-1, 
            0,
            key="lime_sample"
        )
        
        # Display selected sample
        st.write("Selected sample features:")
        X = sample_data.drop('isFraud', axis=1, errors='ignore')
        selected_sample = X.iloc[sample_index:sample_index+1]
        st.dataframe(selected_sample)
        
        # Predict on the selected sample
        prediction = loaded_model.predict_proba(selected_sample)[0][1]
        st.write(f"Model prediction: {prediction:.2%} probability of fraud")
        
        # Generate LIME explanation
        try:
            # Initialize LIME explainer
            categorical_features = [i for i, col in enumerate(X.columns) 
                                   if X[col].dtype == 'object' or X[col].nunique() < 10]
            
            lime_explainer = lime_tabular.LimeTabularExplainer(
                X.values,
                feature_names=X.columns.tolist(),
                class_names=["Legitimate", "Fraud"],
                categorical_features=categorical_features,
                mode="classification"
            )
            
            # Generate LIME explanation
            exp = lime_explainer.explain_instance(
                selected_sample.values[0], 
                loaded_model.predict_proba,
                num_features=10
            )
            
            # Plot the explanation
            explanation_list = exp.as_list()
            features, values = zip(*explanation_list)
            
            # Create Plotly bar chart for LIME explanation
            fig = go.Figure()
            colors = ['red' if v < 0 else 'green' for v in values]
            
            fig.add_trace(
                go.Bar(
                    x=values,
                    y=features,
                    orientation='h',
                    marker_color=colors
                )
            )
            
            fig.update_layout(
                title="LIME Explanation",
                xaxis_title="Contribution to Prediction",
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Add interpretation
            st.write("""
            **Interpretation**: 
            - Green bars increase the probability of fraud
            - Red bars decrease the probability of fraud
            - The length of each bar indicates the feature's contribution magnitude
            """)
            
        except Exception as e:
            st.error(f"Error generating LIME explanation: {e}")
            st.error("LIME may require additional data preprocessing for categorical features.")

# Footer with information
st.sidebar.markdown("---")
st.sidebar.info("""
### About
This application implements Explainable AI (XAI) methods to make 
financial fraud detection models more transparent and interpretable.

XAI methods used:
- Feature Importance
- SHAP (Shapley Additive Explanations)
- LIME (Local Interpretable Model-agnostic Explanations)

### Feature Glossary
- **card1**: Card identifier (anonymized)
- **card2**: Card issuer identifier (bin number)
- **card4**: Card network (Visa, Mastercard, etc.)
- **card6**: Card type (Credit/Debit)
- **addr1**: Billing ZIP code (numeric)
- **addr2**: Billing country code (numeric)
- **TransactionAmt**: Transaction amount ($)
- **P_emaildomain**: Purchaser email domain
- **ProductCD**: Product code category
- **DeviceType**: Transaction device (Mobile/Desktop)
                
""")
