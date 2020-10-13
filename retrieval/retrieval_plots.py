import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns 
import numpy as np

import matplotlib

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42


sns.set(style="darkgrid")

df = pd.DataFrame(columns=['R@1', 'R@2', 'R@4', 'R@8', 'R@16'], 
                                data=[[51.1, 64.2, 75.9, 85.7, 92.0],
                                      [60.1, 72.9, 82.9, 89.3, 93.8],
                                      [42.6, 55.0, 66.4, 77.2, 88.8],
                                      [66.1, 76.8, 85.6, 90.6, 94.2]])
df = df.set_index([['ImageNet', 'Ours', '-', '+']])
print(df)

# reorganize df to classic table
df2=df.stack().reset_index()
df2.columns = ['Approach', 'Metric', 'Accuracy']
print(df2)

ax = sns.pointplot(x='Metric', y='Accuracy', hue='Approach', data=df2)
ax.grid(b=True, which='major', linewidth=1.0)
ax.grid(b=True, which='minor', linewidth=0.5)
ax.set(ylim=(39, 100))
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles=handles, labels=labels, loc='lower right')
ax.set(xlabel=' ', ylabel=' ')
y_value=['{:,.0f}'.format(x) + '%' for x in ax.get_yticks()]
ax.set_yticklabels(y_value)
plt.savefig('CUB.pdf')
plt.show()


df = pd.DataFrame(columns=['R@1', 'R@2', 'R@4', 'R@8', 'R@16'], 
                                data=[[40.8, 53.0, 64.9, 76.7, 86.1],
                                      [81.8, 88.7, 93.4, 96.3, 97.9],
                                      [51.5, 63.8, 73.5, 82.4, 96.7],
                                      [84.6, 90.7, 94.1, 96.5, 97.3]])
df = df.set_index([['ImageNet', 'Ours', '-', '+']])
print(df)

# reorganize df to classic table
df2=df.stack().reset_index()
df2.columns = ['Approach', 'Metric', 'Accuracy']
print(df2)

ax = sns.pointplot(x='Metric', y='Accuracy', hue='Approach', data=df2)
ax.grid(b=True, which='major', linewidth=1.0)
ax.grid(b=True, which='minor', linewidth=0.5)
ax.set(ylim=(39, 100))
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles=handles, labels=labels, loc='lower right')
ax.set(xlabel=' ', ylabel=' ')
y_value=['{:,.0f}'.format(x) + '%' for x in ax.get_yticks()]
ax.set_yticklabels(y_value)
plt.savefig('cars.pdf')
plt.show()