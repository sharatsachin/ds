# Matplotlib

## What is Matplotlib and why use it?

Matplotlib is a comprehensive plotting library that provides:
- Publication-quality figures
- Fine-grained control over plot elements
- Multiple output formats (PNG, PDF, SVG, etc.)
- Object-oriented and state-based interfaces
- Extensive customization options
- Integration with NumPy and Pandas

Key advantages:
- Industry standard for Python plotting
- Highly customizable
- Large community and documentation
- Backend flexibility
- Export to multiple formats

## How do you create basic plots?

Basic plotting examples:
```python
import matplotlib.pyplot as plt
import numpy as np

# Line plot
x = np.linspace(0, 10, 100)
plt.plot(x, np.sin(x))
plt.show()

# Scatter plot
plt.scatter(x, np.random.rand(100))
plt.show()

# Bar plot
categories = ['A', 'B', 'C', 'D']
values = [4, 3, 2, 1]
plt.bar(categories, values)
plt.show()

# Histogram
data = np.random.randn(1000)
plt.hist(data, bins=30)
plt.show()

# Multiple plots
plt.plot(x, np.sin(x), label='sin')
plt.plot(x, np.cos(x), label='cos')
plt.legend()
plt.show()
```

## How do you customize plot appearance?

Plot customization examples:
```python
# Figure and axes
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(x, np.sin(x))

# Labels and title
ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
ax.set_title('Simple Plot')

# Line styles and colors
ax.plot(x, np.sin(x), 'r--', linewidth=2, label='sin')
ax.plot(x, np.cos(x), 'b-', linewidth=2, label='cos')

# Markers
ax.plot(x, np.sin(x), 'ro-', markersize=8)

# Grid
ax.grid(True, linestyle='--', alpha=0.7)

# Legend
ax.legend(loc='best')

# Axis limits
ax.set_xlim([0, 10])
ax.set_ylim([-1.5, 1.5])

# Text annotation
ax.text(5, 0.5, 'Sample text', fontsize=12)

# Custom tick labels
ax.set_xticks([0, 2, 4, 6, 8, 10])
ax.set_xticklabels(['0', '2', '4', '6', '8', '10'])

plt.show()
```

## How do you create subplots?

Subplot examples:
```python
# Method 1: subplot grid
fig, axes = plt.subplots(2, 2, figsize=(10, 8))
axes[0, 0].plot(x, np.sin(x))
axes[0, 1].scatter(x, np.random.rand(100))
axes[1, 0].hist(np.random.randn(1000))
axes[1, 1].bar(categories, values)

# Method 2: subplot positions
plt.figure(figsize=(10, 8))
plt.subplot(221)  # 2 rows, 2 cols, position 1
plt.plot(x, np.sin(x))
plt.subplot(222)
plt.scatter(x, np.random.rand(100))
plt.subplot(223)
plt.hist(np.random.randn(1000))
plt.subplot(224)
plt.bar(categories, values)

# Adjust spacing
plt.tight_layout()
plt.show()
```

## How do you create specialized plots?

Advanced plot types:
```python
# Box plot
data = [np.random.normal(0, std, 100) for std in range(1, 4)]
plt.boxplot(data)

# Violin plot
plt.violinplot(data)

# Pie chart
sizes = [30, 20, 30, 20]
labels = ['A', 'B', 'C', 'D']
plt.pie(sizes, labels=labels, autopct='%1.1f%%')

# Heatmap
data = np.random.rand(10, 10)
plt.imshow(data, cmap='hot')
plt.colorbar()

# 3D plot
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
X, Y = np.meshgrid(x, x)
Z = np.sin(np.sqrt(X**2 + Y**2))
ax.plot_surface(X, Y, Z)
```

## How do you save and export plots?

Saving plots:
```python
# Save as PNG
plt.savefig('plot.png', dpi=300, bbox_inches='tight')

# Save as PDF
plt.savefig('plot.pdf', format='pdf', bbox_inches='tight')

# Save as SVG
plt.savefig('plot.svg', format='svg', bbox_inches='tight')

# Save with transparency
plt.savefig('plot.png', transparent=True)
```

## How do you create animations?

Animation examples:
```python
from matplotlib.animation import FuncAnimation

fig, ax = plt.subplots()
xdata, ydata = [], []
ln, = ax.plot([], [], 'ro')

def init():
    ax.set_xlim(0, 2*np.pi)
    ax.set_ylim(-1, 1)
    return ln,

def update(frame):
    xdata.append(frame)
    ydata.append(np.sin(frame))
    ln.set_data(xdata, ydata)
    return ln,

ani = FuncAnimation(fig, update, frames=np.linspace(0, 2*np.pi, 128),
                   init_func=init, blit=True)
plt.show()
```

## How do you customize styles?

Style customization:
```python
# Use built-in style
plt.style.use('seaborn')

# Custom style dictionary
custom_style = {
    'axes.facecolor': 'lightgray',
    'axes.grid': True,
    'grid.color': 'white',
    'grid.linestyle': '--',
    'font.size': 12,
    'figure.figsize': [10, 6]
}

# Apply custom style
with plt.style.context(custom_style):
    plt.plot(x, np.sin(x))

# Multiple styles
plt.style.use(['seaborn', 'dark_background'])
```

## How do you handle color maps?

Colormap usage:
```python
# Built-in colormaps
plt.imshow(data, cmap='viridis')
plt.colorbar()

# Custom colormap
from matplotlib.colors import LinearSegmentedColormap
colors = ['red', 'yellow', 'green']
n_bins = 100
cmap = LinearSegmentedColormap.from_list('custom', colors, N=n_bins)

# Apply custom colormap
plt.imshow(data, cmap=cmap)

# Normalize colormap
from matplotlib.colors import Normalize
norm = Normalize(vmin=data.min(), vmax=data.max())
plt.imshow(data, cmap='viridis', norm=norm)
```

## How do you create custom legends?

Legend customization:
```python
# Custom legend entries
from matplotlib.lines import Line2D
custom_lines = [Line2D([0], [0], color='red', lw=2),
               Line2D([0], [0], color='blue', lw=2)]
plt.legend(custom_lines, ['Line 1', 'Line 2'])

# Legend with multiple columns
plt.legend(ncol=2)

# Custom legend location
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

# Legend outside plot
plt.legend(bbox_to_anchor=(1.04, 0.5), loc='center left')
```