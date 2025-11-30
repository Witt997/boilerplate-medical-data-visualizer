import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# 1. Import the data
df = pd.read_csv('medical_examination.csv')

# 2. Add an overweight column
# BMI = weight(kg) / (height(m))^2
df['overweight'] = (df['weight'] / ((df['height']/100) ** 2) > 25).astype(int)

# 3. Normalize data: 0 always good, 1 always bad
df['cholesterol'] = (df['cholesterol'] > 1).astype(int)
df['gluc'] = (df['gluc'] > 1).astype(int)

# 4. Draw the Categorical Plot
def draw_cat_plot():
    # Melt the DataFrame to long format with the correct order
    df_cat = pd.melt(
        df,
        id_vars=['cardio'],
        value_vars=['active', 'alco', 'cholesterol', 'gluc', 'overweight', 'smoke']
    )

    # Draw the catplot using seaborn countplot
    g = sns.catplot(
        x='variable',
        hue='value',
        col='cardio',
        data=df_cat,
        kind='count',
        order=['active', 'alco', 'cholesterol', 'gluc', 'overweight', 'smoke'],
        hue_order=[0,1]
    )

    # Set y-axis label to 'total' for all subplots
    g.set_axis_labels("variable", "total")

    # Get the figure
    fig = g.fig

    # Save the figure
    fig.savefig('catplot.png')
    return fig


# 10. Draw the Heat Map
def draw_heat_map():
    # 11. Clean the data
    df_heat = df[
        (df['ap_lo'] <= df['ap_hi']) &
        (df['height'] >= df['height'].quantile(0.025)) &
        (df['height'] <= df['height'].quantile(0.975)) &
        (df['weight'] >= df['weight'].quantile(0.025)) &
        (df['weight'] <= df['weight'].quantile(0.975))
    ]

    # 12. Calculate correlation matrix
    corr = df_heat.corr()

    # 13. Generate a mask for the upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool))

    # 14. Set up the matplotlib figure
    fig, ax = plt.subplots(figsize=(12, 10))

    # 15. Draw the heatmap
    sns.heatmap(corr, mask=mask, annot=True, fmt=".1f", center=0, square=True, linewidths=.5, cbar_kws={"shrink": .5}, ax=ax)

    # 16. Save figure
    fig.savefig('heatmap.png')
    return fig
