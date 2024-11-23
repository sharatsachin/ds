# Seaborn

## What is Seaborn and why use it?

Seaborn is a statistical visualization library that provides:
- High-level interface for statistical graphics
- Beautiful default styles
- Built-in themes
- Integration with Pandas DataFrames
- Statistical estimation and visualization
- Complex plot types with simple commands

Key advantages:
- Built on top of Matplotlib
- Attractive default aesthetics
- Statistical plotting functions
- Automatic handling of Pandas data
- Built-in themes and color palettes

## How do you create basic plots?

Basic plotting examples:
```python
import seaborn as sns
import pandas as pd
import numpy as np

# Sample data
tips = sns.load_dataset('tips')
flights = sns.load_dataset('flights')

# Scatter plot
sns.scatterplot(data=tips, x='total_bill', y='tip')

# Line plot
sns.lineplot(data=flights, x='year', y='passengers')

# Bar plot
sns.barplot(data=tips, x='day', y='total_bill')

# Count plot
sns.countplot(data=tips, x='day')

# Box plot
sns.boxplot(data=tips, x='day', y='total_bill')

# Violin plot
sns.violinplot(data=tips, x='day', y='total_bill')

# Strip plot
sns.stripplot(data=tips, x='day', y='total_bill')
```

## How do you create statistical plots?

Statistical visualization:
```python
# Regression plot
sns.regplot(data=tips, x='total_bill', y='tip')

# Residual plot
sns.residplot(data=tips, x='total_bill', y='tip')

# Joint distribution
sns.jointplot(data=tips, x='total_bill', y='tip', kind='reg')

# Distribution plot
sns.displot(data=tips, x='total_bill', kind='kde')

# Pair plot
sns.pairplot(data=tips)

# Correlation heatmap
sns.heatmap(tips.corr(), annot=True, cmap='coolwarm')

# Cluster map
sns.clustermap(tips.corr(), cmap='coolwarm')
```

## How do you customize plot appearance?

Plot customization:
```python
# Set style
sns.set_style('whitegrid')

# Set context
sns.set_context('paper')  # or 'notebook', 'talk', 'poster'

# Set palette
sns.set_palette('husl')

# Custom figure size
plt.figure(figsize=(10, 6))
sns.scatterplot(data=tips, x='total_bill', y='tip')

# Add title and labels
g = sns.scatterplot(data=tips, x='total_bill', y='tip')
g.set_title('Tips vs Total Bill')
g.set_xlabel('Total Bill ($)')
g.set_ylabel('Tip ($)')

# Rotate x-axis labels
plt.xticks(rotation=45)

# Adjust plot spacing
plt.tight_layout()
```

## How do you create advanced visualizations?

Advanced plotting:
```python
# FacetGrid
g = sns.FacetGrid(tips, col='time', row='smoker')
g.map(sns.scatterplot, 'total_bill', 'tip')

# LMPlot
sns.lmplot(data=tips, x='total_bill', y='tip', 
           col='time', row='smoker')

# Complex categorical plots
sns.catplot(data=tips, x='day', y='total_bill', 
            kind='violin', hue='sex', split=True)

# Joint distributions with different kinds
sns.jointplot(data=tips, x='total_bill', y='tip', 
             kind='hex')

# Multiple distributions
sns.displot(data=tips, x='total_bill', 
           hue='time', multiple='stack')
```

## How do you handle color palettes?

Color customization:
```python
# Built-in palettes
sns.set_palette('Set2')

# Custom color palette
colors = ['#FF0000', '#00FF00', '#0000FF']
sns.set_palette(sns.color_palette(colors))

# Continuous color palette
sns.color_palette('rocket', as_cmap=True)

# Diverging palette
sns.diverging_palette(240, 10, n=9)

# Custom colormap for heatmap
sns.heatmap(data, cmap='coolwarm')

# Color palette with specific number of colors
sns.color_palette('husl', n_colors=5)
```

## How do you create multi-panel figures?

Multi-panel plotting:
```python
# Figure-level interface
g = sns.FacetGrid(tips, col='time', row='smoker')
g.map(sns.histplot, 'total_bill')

# Multiple plots with different scales
g = sns.FacetGrid(tips, col='time')
g.map_dataframe(sns.scatterplot, 'total_bill', 'tip')
g.add_legend()

# Complex layouts
g = sns.PairGrid(tips)
g.map_diag(sns.histplot)
g.map_offdiag(sns.scatterplot)

# Conditional plots
sns.jointplot(data=tips, x='total_bill', y='tip', 
             hue='time')
```

## How do you handle large datasets?

Large dataset visualization:
```python
# Use swarmplot for dense categorical data
sns.swarmplot(data=tips, x='day', y='total_bill', 
              size=3)

# Use hexbin for dense scatter plots
sns.jointplot(data=tips, x='total_bill', y='tip', 
             kind='hex')

# KDE plots for density estimation
sns.kdeplot(data=tips, x='total_bill', y='tip')

# Violin plots with inner points
sns.violinplot(data=tips, x='day', y='total_bill', 
               inner='points')
```

## How do you customize statistical estimates?

Statistical customization:
```python
# Custom confidence intervals
sns.regplot(data=tips, x='total_bill', y='tip', 
            ci=95)

# Bootstrap resampling
sns.lmplot(data=tips, x='total_bill', y='tip', 
           n_boot=1000)

# Kernel bandwidth adjustment
sns.kdeplot(data=tips['total_bill'], bw_adjust=0.5)

# Custom statistical functions
sns.barplot(data=tips, x='day', y='total_bill', 
            estimator=np.median)
```