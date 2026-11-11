import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import os

os.makedirs("outputs/charts", exist_ok=True)
os.makedirs("data", exist_ok=True)

np.random.seed(42)
n_control = 15000
n_treatment = 15000

control_converted = np.random.binomial(1, 0.10, n_control)
treatment_converted = np.random.binomial(1, 0.113, n_treatment)

df_control = pd.DataFrame({'user_id': range(1, n_control + 1), 'group': 'A', 'converted': control_converted})
df_treatment = pd.DataFrame({'user_id': range(n_control + 1, n_control + n_treatment + 1), 'group': 'B', 'converted': treatment_converted})
df = pd.concat([df_control, df_treatment], ignore_index=True)
df.to_csv("data/checkout_data.csv", index=False)

obs_A = sum(df['group'] == 'A')
obs_B = sum(df['group'] == 'B')
chi2, srm_p_value = stats.chisquare([obs_A, obs_B], f_exp=[15000, 15000])

conv_A = df[df['group'] == 'A']['converted'].mean()
conv_B = df[df['group'] == 'B']['converted'].mean()
diff = conv_B - conv_A

se = np.sqrt((conv_A * (1 - conv_A) / n_control) + (conv_B * (1 - conv_B) / n_treatment))
z_stat = diff / se
p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
ci_lower = diff - 1.96 * se
ci_upper = diff + 1.96 * se

plt.figure(figsize=(7, 4))
plt.errorbar(x=['Lift (B - A)'], y=[diff * 100], 
             yerr=[[(diff - ci_lower) * 100], [(ci_upper - diff) * 100]], 
             fmt='o', color='#008080', ecolor='#008080', elinewidth=3, capsize=8, markersize=8)
plt.axhline(0, color='gray', linestyle='--')
plt.title('A/B Test Result: One-Step Checkout Lift')
plt.ylabel('Percentage Point Difference (%)')
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig("outputs/charts/conversion_rate_ci.png")
