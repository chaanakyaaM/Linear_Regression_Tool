import pandas as pd
import streamlit as st
import plotly.express as px
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error

# Set up the title of the app
st.title("Linear Regression ")
st.write("Linear Regression aims to model the relationship between a dependent variable y (target) and one or more independent variables X (features) by fitting a linear equation to the observed data.")
st.write("**NOTE:** Make sure your data is clean and properly formatted before uploading.")

# --- Session State Initialization ---
# Initialize session state variables if they don't exist
if "dataset" not in st.session_state:
    st.session_state.dataset = None
if "cleaned_dataset" not in st.session_state: # New state for cleaned data
    st.session_state.cleaned_dataset = None
if "model" not in st.session_state:
    st.session_state.model = None
if "x_test" not in st.session_state:
    st.session_state.x_test = None
if "y_test" not in st.session_state:
    st.session_state.y_test = None
if "x" not in st.session_state:
    st.session_state.x = None
if "y" not in st.session_state: 
    st.session_state.y = None
if "selected_features" not in st.session_state: 
    st.session_state.selected_features = []
if "y_pred" not in st.session_state: 
    st.session_state.y_pred = None
if "x_train" not in st.session_state: 
    st.session_state.x_train = None
if "y_train" not in st.session_state: 
    st.session_state.y_train = None

st.session_state.target_variable = None

# --- File Upload  ---
uploaded_file = st.file_uploader('Upload CSV file', type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success("File uploaded successfully!")
    st.write("Dataset Preview:")
    st.session_state.dataset = df.copy() 
    st.session_state.cleaned_dataset = df.copy() 
    st.write(df.head())


    with st.expander("Dataset Info"):
        st.write(f"Shape: {st.session_state.dataset.shape}")
        st.write(st.session_state.dataset.describe())
        st.write("Data types:")
        st.write(st.session_state.dataset.dtypes)
        
        # Show missing values info
        missing_values = st.session_state.dataset.isnull().sum()
        if missing_values.sum() > 0:
            st.write("Missing values per column:")
            st.write(missing_values[missing_values > 0])
        else:
            st.write("No missing values found in the dataset.")

    with st.expander('Feature and Target Selection'):
        if st.session_state.cleaned_dataset is not None:
            # Only show numerical columns for feature selection
            numerical_cols = st.session_state.cleaned_dataset.select_dtypes(include=['number']).columns.tolist()

            # Multiselect for features
            st.session_state.selected_features = st.multiselect(
                "Select features for the model:",
                numerical_cols,
                default=st.session_state.selected_features if st.session_state.selected_features else []
            )
            
            # Selectbox for target variable
            if numerical_cols:
                default_target_index = 0
                if st.session_state.target_variable in numerical_cols:
                    default_target_index = numerical_cols.index(st.session_state.target_variable)
                
                st.session_state.target_variable = st.selectbox(
                    "Select the target variable:",
                    numerical_cols,
                    index=default_target_index
                )

                if st.session_state.selected_features and st.session_state.target_variable:
                    # Ensure the target variable is not in the features list
                    if st.session_state.target_variable in st.session_state.selected_features:
                        st.session_state.selected_features.remove(st.session_state.target_variable)
                        st.warning(f"'{st.session_state.target_variable}' was removed from features as it is the target variable.")

                    st.write("Selected Features:")
                    st.write(st.session_state.selected_features)
                    st.write("Selected Target Variable:")
                    st.write(st.session_state.target_variable)

                    if st.button("Select Features and Target"):
                        # Set X and y based on selected features and target from the cleaned dataset
                        if st.session_state.selected_features: # Make sure features are selected
                            # Extract the selected columns
                            x_temp = st.session_state.cleaned_dataset[st.session_state.selected_features]
                            y_temp = st.session_state.cleaned_dataset[st.session_state.target_variable]
                            
                            # Check for NaN values in selected features and target
                            x_nan_count = x_temp.isnull().sum().sum()
                            y_nan_count = y_temp.isnull().sum()
                            
                            if x_nan_count > 0 or y_nan_count > 0:
                                st.warning(f"Found {x_nan_count} NaN values in features and {y_nan_count} NaN values in target. Removing rows with NaN values...")
                                
                                # Combine X and y to remove NaN rows together
                                combined_data = pd.concat([x_temp, y_temp], axis=1)
                                combined_clean = combined_data.dropna()
                                
                                if len(combined_clean) == 0:
                                    st.error("No data left after removing NaN values. Please check your data or feature selection.")
                                else:
                                    st.session_state.x = combined_clean[st.session_state.selected_features]
                                    st.session_state.y = combined_clean[st.session_state.target_variable]
                                    st.success(f'Features and target variable selected successfully! Dataset size after cleaning: {len(combined_clean)} rows')
                            else:
                                st.session_state.x = x_temp
                                st.session_state.y = y_temp
                                st.success('Features and target variable selected successfully!')
                        else:
                            st.warning("Please select at least one feature.")

                        # Show previews
                        if st.session_state.x is not None and st.session_state.y is not None:
                            st.write("Features Preview (X):")
                            st.write(st.session_state.x.head())
                            st.write(f"Features shape: {st.session_state.x.shape}")
                            
                            st.write("Target Preview (y):")
                            st.write(st.session_state.y.head())
                            st.write(f"Target shape: {st.session_state.y.shape}")
                            
                            # Check for any remaining NaN values
                            x_nan = st.session_state.x.isnull().sum().sum()
                            y_nan = st.session_state.y.isnull().sum()
                            if x_nan > 0 or y_nan > 0:
                                st.error(f"Warning: Still found {x_nan} NaN values in features and {y_nan} NaN values in target!")
                            else:
                                st.success("✓ No NaN values found in selected data")
                else:
                    st.warning("Please select both features and a target variable.")

    with st.expander("Linear Regression Model Training"):
        with st.expander("Raw dataset"):
            if st.session_state.cleaned_dataset is not None:
                st.write(st.session_state.cleaned_dataset)
        
        if st.session_state.x is not None and st.session_state.y is not None:
            test_size = st.slider("Test Size", min_value=0.1, max_value=0.5, value=0.2, step=0.05, key="test_size_slider")
            st.write(f"Using a test size of: {test_size:.2f}")

            if st.button("Fit Model", key="fit_model_button"):
                st.session_state.model = LinearRegression()
                
                x = st.session_state.x
                y = st.session_state.y

                if x.empty or y.empty:
                    st.error("Features (X) or target variable (y) are empty. Please check your data and selections.")
                else:
                    # Final check for NaN values before training
                    x_nan_count = x.isnull().sum().sum()
                    y_nan_count = y.isnull().sum()
                    
                    if x_nan_count > 0 or y_nan_count > 0:
                        st.error(f"Cannot train model: Found {x_nan_count} NaN values in features and {y_nan_count} NaN values in target. Please clean your data first.")
                    else:
                        try:
                            # Convert to numpy arrays
                            x_values = x.values
                            y_values = y.values
                            
                            # Split the data into train and test sets
                            x_train, x_test, y_train, y_test = train_test_split(
                                x_values, y_values, test_size=test_size, random_state=42
                            )
                            
                            # Store test data for later use
                            st.session_state.x_test = x_test
                            st.session_state.y_test = y_test
                            
                            # Train the model
                            st.session_state.model.fit(x_train, y_train)
                            
                            st.success("Model trained successfully!")
                            st.write("Model coefficients:")
                            if len(st.session_state.selected_features) == 1:
                                st.write(f"{st.session_state.selected_features[0]}: {st.session_state.model.coef_[0]:.4f}")
                            else:
                                coef_df = pd.DataFrame({
                                    'Feature': st.session_state.selected_features,
                                    'Coefficient': st.session_state.model.coef_
                                })
                                st.write(coef_df)
                            
                            st.write("Model intercept:")
                            st.write(f"{st.session_state.model.intercept_:.4f}")
                            
                        except Exception as e:
                            st.error(f"An error occurred during model training: {e}")
        else:
            st.info("Please upload a dataset and select features/target to train the model.")

    with st.expander("Data Visualization and Exploration"):
        if st.session_state.cleaned_dataset is not None:
            numerical_cols = st.session_state.cleaned_dataset.select_dtypes(include=['number']).columns.tolist()
            
            if len(numerical_cols) > 0:
                # Dataset Overview
                st.subheader("📊 Dataset Overview")
                
                
                # Distribution plots
                selected_col = st.selectbox("Select column for distribution plot:", numerical_cols, key="dist_col")
                if selected_col:
                    fig = px.histogram(st.session_state.cleaned_dataset, x=selected_col, 
                                       title=f"Distribution of {selected_col}", marginal="box")
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)
                
                # Summary statistics
                st.write("**Summary Statistics:**")
                summary_stats = st.session_state.cleaned_dataset[numerical_cols].describe()
                st.dataframe(summary_stats, use_container_width=True)
                
            else:
                st.warning("No numerical columns found for visualization.")

            if st.session_state.model and st.session_state.x_test is not None and st.session_state.y_test is not None:
                if st.button("Make Predictions and Evaluate", key="predict_button"):
                    model = st.session_state.model
                    x_test = st.session_state.x_test
                    y_test = st.session_state.y_test

                    try:
                        predict = model.predict(x_test)             
                        st.subheader("Prediction Results:")
                        results_df = pd.DataFrame({
                            'Actual': y_test.flatten(),
                            'Predicted': predict.flatten()
                        })
                        results_df['Difference'] = results_df['Actual'] - results_df['Predicted']
                        results_df['Abs_Difference'] = abs(results_df['Difference'])
                        
                        st.write("Sample predictions (first 10 rows):")
                        st.write(results_df.head(10))

                        st.subheader("Model Metrics:")
                        r2 = r2_score(y_test, predict)
                        mae = mean_absolute_error(y_test, predict)
                        model_score = model.score(x_test, y_test)
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("R-squared (R²)", f"{r2:.4f}")
                        with col2:
                            st.metric("Mean Absolute Error", f"{mae:.4f}")
                        with col3:
                            st.metric("Model Score", f"{model_score:.4f}")
                        
                        # Additional metrics
                        st.write("**Interpretation:**")
                        st.write(f"- R² score of {r2:.4f} means the model explains {r2*100:.2f}% of the variance in the target variable")
                        st.write(f"- Mean Absolute Error of {mae:.4f} indicates the average prediction error")
                        
                        if r2 > 0.7:
                            st.success("Good model performance! (R² > 0.7)")
                        elif r2 > 0.5:
                            st.warning("Moderate model performance (0.5 < R² ≤ 0.7)")
                        else:
                            st.error("Poor model performance (R² ≤ 0.5)")

                        # --- Linear Regression Visualization Graph and Line ---
                        st.subheader("🎯 Actual vs. Predicted Values with Regression Line")
                        if not results_df.empty:
                            fig_reg = px.scatter(results_df, x='Actual', y='Predicted',
                                                title='Actual vs. Predicted Values with OLS Regression Line',
                                                trendline='ols', # Add the Ordinary Least Squares regression line
                                                labels={'Actual': 'Actual Values', 'Predicted': 'Predicted Values'})
                            
                            # Add a perfect prediction line (y=x) for reference
                            max_val = max(results_df['Actual'].max(), results_df['Predicted'].max())
                            min_val = min(results_df['Actual'].min(), results_df['Predicted'].min())
                            fig_reg.add_shape(type="line", line=dict(dash='dash', color='red'),
                                            x0=min_val, y0=min_val, x1=max_val, y1=max_val)
                            
                            fig_reg.update_layout(height=500, width=700)
                            st.plotly_chart(fig_reg, use_container_width=True)
                        else:
                            st.warning("No prediction results available to plot the regression graph.")
                        
                    except Exception as e:
                        st.error(f"An error occurred during prediction or evaluation: {e}")
            else:
                st.info("Please train the model first to see predictions and evaluations.")

            plt.plot(st.session_state.x,st.session_state.y, label="Data Points")
            plt.xlabel("Features")
            plt.ylabel("Target Variable")
            plt.show()

elif uploaded_file is None and st.session_state.dataset is None:
    st.info("Please upload a CSV file to get started!")

