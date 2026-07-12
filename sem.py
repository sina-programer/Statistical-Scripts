from semopy import Model, semplot, calc_stats  # pip install semopy
import pandas as pd
import numpy as np

np.random.seed(42)
n = 200

# path coefficients
a, b, c = 0.7, 0.5, 0.2

X = np.random.normal(0, 1, n)  # exogenous latent: SocialStatus
M = a * X + np.random.normal(0, 0.3, n) # mediator: Health
Y = b * M + c * X + np.random.normal(0, 0.3, n) # endogenous: Wellbeing

df = pd.DataFrame({
    # Indicators for X (SocialStatus)
    'income': X + np.random.normal(0, 0.4, n),
    'education': 0.8*X + np.random.normal(0, 0.4, n),
    'occupation': 0.9*X + np.random.normal(0, 0.4, n),

    # Indicators for M (Health)
    'bmi': -0.6*M + np.random.normal(0, 0.4, n),
    'bp': 0.7*M + np.random.normal(0, 0.4, n),
    'chol': 0.8*M + np.random.normal(0, 0.4, n),

    # Indicators for Y (Wellbeing)
    'ls': Y + np.random.normal(0, 0.4, n),
    'hap': 0.9*Y + np.random.normal(0, 0.4, n),
    'pa': 0.8*Y + np.random.normal(0, 0.4, n)
})

model_desc = """
# Measurement Model
SocialStatus =~ income + 1*education + p*occupation
Health       =~ bmi + bp + chol
Wellbeing    =~ ls + hap + pa

# Structural Model (regression)
Health       ~ a*SocialStatus  # 'Health' is regressed on 'SocialStatus'
Wellbeing    ~ b*Health + c*SocialStatus

# Residual Correlation
Health ~~ Wellbeing
"""

model = Model(model_desc)
result = model.fit(df)

print("Model Fit Result:")
print(result)

print('\nEstimates:')
estimates = model.inspect()
print(estimates)

print("\nModel Fit Statistics:")
stats = calc_stats(model)
print(stats.T)

semplot(model, 'model.png')
