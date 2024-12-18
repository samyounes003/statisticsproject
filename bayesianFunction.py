from sklearn.linear_model import LassoCV
import numpy as np
import pymc as pm
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import arviz as az
from sklearn.decomposition import PCA


def df_basic_process(data:pd.DataFrame):
    data = data[(data.priceCash > 300000) & (data.priceCash < 20000000)] # filter out really low and high prices
    data['priceCash'] = data.priceCash / 7.45
    important_features = ['priceCash', 'coordinates.lat', 'coordinates.lon', 'energyLabel', 'housingArea', 'lotArea', 'monthlyExpense', 'numberOfRooms', 'perAreaPrice', 'timeOnMarket.current.days', 'yearBuilt']
    data = data[[i for i in data.columns if i in important_features]]
    return data



# Lasso regression minimizes the residual sum of squares (RSS), which assumes the errors (residuals) follow a Gaussian (normal) distribution. This assumption often holds well when:
# The target variable itself (e.g., housing prices) is normally distributed.
# Or, the target has been transformed to approximate normality (e.g., log transformation).
# In your case, you log-transformed the prices (PriceLogged) to make the distribution more normal. This transformation aligns well with Lasso’s assumptions, as it reduces skewness and stabilizes variance, leading to:
# Better predictions.
# A model where the coefficients more accurately reflect the relationships between features and the target.

def preprocess_data(X: pd.DataFrame, use_pca: bool = False, n_components: int = 10):
    # Identify numeric and categorical columns
    numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns
    categorical_cols = X.select_dtypes(include=['object']).columns

    # Define preprocessing pipelines
    numeric_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False))
    ])

    # Combine pipelines using ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_pipeline, numeric_cols),
            ('cat', categorical_pipeline, categorical_cols)
        ]
    )

    # Fit and transform the data
    X_transformed = preprocessor.fit_transform(X)

    if use_pca:
        # Apply PCA to reduce dimensions
        pca = PCA(n_components=n_components)
        X_pca = pca.fit_transform(X_transformed)

        # Create column names for PCA components
        pca_features = [f"PCA_Component_{i+1}" for i in range(n_components)]

        # Convert PCA-transformed data to a DataFrame
        X_df = pd.DataFrame(X_pca, columns=pca_features)
    else:
        # Dynamically extract feature names after transformation
        numeric_features = list(numeric_cols)  # Numeric features remain as is
        categorical_features = preprocessor.named_transformers_['cat']['onehot'].get_feature_names_out(categorical_cols)

        # Combine all features
        processed_features = numeric_features + list(categorical_features)

        # Truncate features dynamically to align with X_transformed
        processed_features = processed_features[:X_transformed.shape[1]]

        # Convert transformed data to a DataFrame
        X_df = pd.DataFrame(X_transformed, columns=processed_features)

    return X_df


# find the best features using Lasso
def find_best_features_with_lasso(X_df:pd.DataFrame, y_df:pd.DataFrame):
    # Train Lasso model with cross-validation
    lasso = LassoCV(cv=5, random_state=42).fit(X_df, y_df)

    # Get feature importance
    lasso_coefficients = pd.Series(lasso.coef_, index=X_df.columns)
    important_features_to_use = lasso_coefficients[lasso_coefficients != 0]

    # sort the features in descending order of importance
    important_features_to_use = important_features_to_use.sort_values(ascending=False)
    # print("Lasso Selected Features:\n", important_features_to_use)

    important_features = list(important_features_to_use.index)
    return important_features


# get the posterior probability
# get the priors: coefficients and intercept (both normal distribution with both having mu=0 and sigma=10)
# make the likelihood (normal (mu, sigma)):
#       mu = use the priors, combined with the features to make linear functions for the mu paramter of the likelihood distribution
#       sigma = HalfNormal???
# use mcmc to calculate the posterior distribution

# coefficients ───┐
#                 │   ---> mu = dot(X, coefficients) + intercept
# intercept   ───┘               │
#                                 ↓
# sigma ──────────────────────────┼──> price_obs ~ Normal(mu, sigma)
#                                 │
#                            Observed Data: y_train

def get_bayesian_posterior_distribution(X_train:pd.DataFrame, y_train:pd.DataFrame, sigma:int=10):
    # Define the Bayesian model
    with pm.Model() as housing_model:
        # Priors for coefficients
                
        ### LAPLACE FOR PRIORS
        # coefficients = pm.Laplace('coefficients', mu=0, b=1, shape=X_train.shape[1])

        ### NORMAL FOR PRIORS
        coefficients = pm.Normal('coefficients', mu=0, sigma=sigma, shape=X_train.shape[1])
        intercept = pm.Normal('Intercept', mu=0, sigma=sigma)

        ### UNIFORM FOR PRIORS
        # coefficients = pm.Uniform('coefficients', lower=-1e5, upper=1e5, shape=X_train.shape[1])
        # intercept = pm.Uniform('Intercept', lower=-1e5, upper=1e5)


        # Define the linear model
        mu = pm.math.dot(X_train.values, coefficients) + intercept


        # log_sigma = pm.Deterministic(
        #     'log_sigma', 
        #     pm.math.dot(X_train.values, pm.Normal('sigma_coefficients', mu=0, sigma=10, shape=X_train.shape[1])) + 
        #     pm.Normal('sigma_intercept', mu=0, sigma=10)
        # )
        # sigma = pm.Exponential('sigma', lam=pm.math.exp(log_sigma))  # Exponential to keep sigma positive


        # # Likelihood function        
        sigma = pm.HalfNormal('sigma', sigma=1)   
        # #"Given the parameters μ and σ, how likely are the observed values y train to occur?"  
        ### UNIFORM FOR LIKELIHOOD
        price_obs = pm.Normal('Price', mu=mu, sigma=sigma, observed=y_train.values) # The observed=y_train.values part in price_obs tells PyMC: These are the actual observed values for the target variable.

        # ### STUDENT T FOR LIKELIHOOD - works better with outliers
        # nu = pm.Exponential('nu', 1/30)  # Degrees of freedom for heavy tails
        # price_obs = pm.StudentT('Price', mu=mu, sigma=sigma, nu=nu, observed=y_train)


        # Sampling using MCMC
        # Identifies all the stochastic variables (like coefficients, intercept, sigma) and the likelihood (price_obs) in the model.
        # Initializes the posterior sampling process using algorithms like NUTS (No-U-Turn Sampler, a variant of Hamiltonian Monte Carlo).
        # Iteratively samples from the posterior distributions of the parameters, ensuring that the samples are consistent with the prior distributions and the likelihood of the observed data.        
        
        # How Does PyMC Know What to Sample?
        # Priors: Any random variable (e.g., pm.Normal, pm.HalfNormal) defined inside the pm.Model() context is recognized as a prior.
        # Likelihood: Any variable defined with observed=... is treated as the likelihood and connects the observed data to the priors.
        # Relationships: Deterministic relationships (like mu = dot(...) + intercept) are automatically included in the model graph.        
        trace = pm.sample(1000, tune=1000, random_seed=42, cores=1)

    # Summarize posterior distributions
    summary = pm.summary(trace)
    return summary, trace, housing_model


# use the coefficients to determine the most important features (removing the unnecessary features)
def features_importance_analysis_coeffs(trace, X_train:pd.DataFrame):
    # Get posterior summaries for coefficients
    posterior_summary = az.summary(trace, var_names=["coefficients"])
    posterior_summary

    # get the coefficients which are greater than 0
    not_0_coeffs = np.where(np.array(list(posterior_summary['mean'])) > 0)[0]
    important_features = [list(X_train.columns)[i] for i in not_0_coeffs]

    # add the feature names to the df
    best_features_coeffs = posterior_summary.iloc[not_0_coeffs]
    best_features_coeffs['feature_names'] = important_features

    # sort the features on ascending order of importance
    best_features_coeffs = best_features_coeffs.sort_values(by='mean', ascending=False)
    return best_features_coeffs