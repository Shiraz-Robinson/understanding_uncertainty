# %%
import pandas as pd
import seaborn as sns
import numpy as np

import statsmodels.api as sm
import pymc as pm

# Load Ames housing data
df = pd.read_csv('./data/ames_prices.csv',low_memory=False)
df['age'] = df['Yr.Sold']-df['Year.Built']
df['age_sq'] = df['age']**2

numeric_vars = ['price', 'area', 'Lot.Area', 'age', 'age_sq']
X_num = df.loc[:,numeric_vars]

# Create dummy variables/fixed effects/one hot encode:
ac = pd.get_dummies(df['Central.Air'],drop_first=True, dtype=int)
fireplaces = pd.get_dummies(df['Fireplaces'],drop_first=True, dtype=int)
type = pd.get_dummies(df['Bldg.Type'],drop_first=True, dtype=int)
style = pd.get_dummies(df['House.Style'],drop_first=True, dtype=int)
foundation = pd.get_dummies(df['Foundation'],drop_first=True, dtype=int)
quality = pd.get_dummies(df['Overall.Qual'],drop_first=True, dtype=int)
#
all = pd.concat( [X_num, ac, type, style, quality], axis=1)
print(all.shape)
all = all.dropna()
print(all.shape)

y = np.log(all['price'])
x = all['age']


# %%

## Bayesian single linear model:

with pm.Model() as model:

    # Priors for intercept and slope:
    beta0 = pm.Normal("beta0", mu=0, sigma=10)
    beta1 = pm.Normal("beta1", mu=0, sigma=10)

    # Prior for standard deviation of shocks:
    sigma = pm.HalfNormal("sigma", sigma=1)

    # Expected value of log-price
    mu = beta0 + beta1 * x

    # Likelihood:
    y_obs = pm.Normal("y_obs", mu=mu, sigma=sigma, observed=y)

    # Sample posterior
    posterior = pm.sample(1000, tune=1000, target_accept=0.9)


pm.plot_posterior(posterior, var_names=["beta0", "beta1", "sigma"]) # Posterior densities
pm.summary(posterior, var_names=["beta0", "beta1", "sigma"])
#pm.plot_trace(posterior, var_names=["beta0", "beta1", "sigma"]);


# %%

# Single linear OLS model:
X = x
X = sm.add_constant(X)
model = sm.OLS(y, X) # Linear regression object
reg = model.fit() # Fit model
print(reg.summary()) # Summary table

# %%

## Full Bayesian model:

# Build design matrix for multiple regression
X = all.drop(columns=["price"])   # drop target, keep predictors
X = sm.add_constant(X)

# Convert to numpy for PyMC
X_np = X.to_numpy()
y_np = y.to_numpy()

n, k = X_np.shape

with pm.Model() as model:

    # Prior for regression coefficients
    beta = pm.Normal("beta", mu=0, sigma=10, shape=k)

    # Prior for standard deviation
    sigma = pm.HalfNormal("sigma", sigma=1)

    # Conditional expectation function
    mu = pm.math.dot(X_np, beta)

    # Likelihood
    y_obs = pm.Normal("y_obs", mu=mu, sigma=sigma, observed=y_np)

    # Sample the posterior
    posterior = pm.sample(1500, tune=1500, target_accept=0.9)

## Analyze posterior:
pm.plot_posterior(posterior, var_names=["beta"]);
pm.summary(posterior, var_names=["beta", "sigma"])
 
# %%

## OLS Comparison:

reg = sm.OLS(y_np, X_np).fit()
print(reg.summary())

# %%
