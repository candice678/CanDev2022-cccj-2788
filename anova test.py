
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import copy
import warnings
warnings.filterwarnings("ignore")

# Get plotting dataframe
df = pd.read_csv("./INCIDENTS.csv")
sub_df = copy.deepcopy(df[["ASSIGNED_GROUP", "BUSINESS_COMPLETION_HRS"]])
sub_df = sub_df.sort_values(["ASSIGNED_GROUP"])
sub_df.columns = ["g", "x"]

# Dealing with missing data, simply drop missing value.
sub_df = sub_df.dropna()

time_df = sub_df.groupby("g").agg('mean').sort_values(["x"])

sub_df.describe()

sns.set_theme(style="whitegrid")
sns.boxplot(x=sub_df["x"])

"""这里发现存在异常值，需要进行处理。"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats
import copy

df

val_df = df[['TICKET_NMBR','PARENT_SERVICE','BUSINESS_COMPLETION_HRS']]
val_df

his_df = pd.read_csv("./INCIDENT_HISTORY.csv")
his_df = his_df[['ticket_nmbr','TIME_IN_STATUS_HRS']]
his_df.rename(columns={'ticket_nmbr':'TICKET_NMBR'},inplace=True)

his_df

his_df = his_df.groupby("TICKET_NMBR")['TIME_IN_STATUS_HRS'].sum()
val_df = val_df.merge(his_df,how='inner',on='TICKET_NMBR')
val_df["measure"] = val_df["BUSINESS_COMPLETION_HRS"] - val_df["TIME_IN_STATUS_HRS"]
val_df = val_df.sort_values(["measure"])
val_df.dropna(inplace=True)
val_df

val_df.describe()

"""We find it more accurate to substitue BUSINESS_COMPLETION_HRS with TIME_IN_STATUS_HRS"""

sns.displot(val_df, x="TIME_IN_STATUS_HRS", kind="kde")

sns.boxplot(x=val_df["TIME_IN_STATUS_HRS"])

sns.displot(val_df, x="TIME_IN_STATUS_HRS", kind="ecdf")

"""根据上图可知，2000可以包含绝大部分数据"""

df_2000 = val_df[val_df['TIME_IN_STATUS_HRS']<2000]
sns.displot(df_2000, x="TIME_IN_STATUS_HRS", kind="kde")

sns.boxplot(x=df_2000["TIME_IN_STATUS_HRS"])

sns.displot(df_2000, x="TIME_IN_STATUS_HRS", kind="ecdf")

"""进一步降至500"""

df_500 = val_df[val_df['TIME_IN_STATUS_HRS']<500]
sns.displot(df_500, x="TIME_IN_STATUS_HRS", kind="kde")

sns.boxplot(x=df_500["TIME_IN_STATUS_HRS"])

sns.displot(df_500, x="TIME_IN_STATUS_HRS", kind="ecdf")

df_50 = val_df[val_df['TIME_IN_STATUS_HRS']<50]
df_10_50 = df_50[df_50['TIME_IN_STATUS_HRS']>10]
sns.displot(df_10_50, x="TIME_IN_STATUS_HRS")

df_50.count()

"""小于50的数据一共有18万条，包含大部分情况，选用10~50的数据比较合理。"""

sns.boxplot(x=df_10_50["TIME_IN_STATUS_HRS"])

sns.displot(df_10_50, x="TIME_IN_STATUS_HRS", kind="ecdf")

"""下面开始画图"""

final_df = df_10_50
final_df = final_df.groupby("PARENT_SERVICE")
final_df

df_10_50

sub_df = copy.deepcopy(df_10_50[["PARENT_SERVICE", "TIME_IN_STATUS_HRS"]])
sub_df = sub_df.sort_values(["PARENT_SERVICE"])
sub_df.columns = ["g", "x"]

plotting_df = sub_df
sns.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})
pal = sns.cubehelix_palette(11, rot=-.25, light=.7)
g = sns.FacetGrid(plotting_df, row="g", hue="g", aspect=12, height=.5, palette=pal)
g.map(sns.kdeplot, "x",
      bw_adjust=.5, clip_on=False,
      fill=True, alpha=1, linewidth=1.5)
g.map(sns.kdeplot, "x", clip_on=False, color="w", lw=2, bw_adjust=.5)
g.map(plt.axhline, y=0, lw=2, clip_on=False)

def label(x, color, label):
    ax = plt.gca()
    ax.text(0, .2, label, fontweight="bold", color="black",
            ha="left", va="center", transform=ax.transAxes)

g.map(label, "x")
g.fig.subplots_adjust(hspace=-.25)
g.set_titles("")
g.set_xlabels("text", fontsize=10, fontweight="bold", color=pal[-3])
g.set(yticks=[], ylabel="")
g.despine(bottom=True, left=True)

sub_df['g'].unique()

sel_df = sub_df[sub_df['g'].isin(['Classified Infrastructure','Government of Canada Managed Security Service',
                  'Internal Credential Management (ICM)','Managed Secure File Transfer',
                  'Satellite','Workplace Technology Services',
                  'Middleware','Midrange','GC WAN','Other Activities','High-performance Computing'])]

sel_df

plotting_df = sel_df
sns.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})
pal = sns.cubehelix_palette(10, rot=-.25, light=.7)
g = sns.FacetGrid(plotting_df, row="g", hue="g", aspect=12, height=.7, palette=pal)
g.map(sns.kdeplot, "x",
      bw_adjust=.5, clip_on=False,
      fill=True, alpha=1, linewidth=1.5)
g.map(sns.kdeplot, "x", clip_on=False, color="w", lw=2, bw_adjust=.5)
g.refline(y=0, linewidth=2, linestyle="-", color=None, clip_on=False)
# g.map(plt.axhline, y=0, lw=2, clip_on=False)

def label(x, color, label):
    ax = plt.gca()
    ax.text(0, .2, label, fontweight="bold", color="black",
            ha="left", va="center", transform=ax.transAxes)

g.map(label, "x")
g.fig.subplots_adjust(hspace=-.25)
g.set_titles("")
g.set_xlabels("Completion Times Hrs", fontsize=10, fontweight="bold", color=pal[-3])
g.set(yticks=[], ylabel="")
g.despine(bottom=True, left=True)
g.savefig('./test.png')

"""由图可知，该分布不属于正态分布，因此不满足ANOVA Test中的正态分布假定，故选用非参数检验方法。下面使用Kruskal-Wallis Test进行假设检验。

$$
H=(N-1) \frac{\sum_{i=1}^{g} n_{i}\left(\bar{r}_{i}-\bar{r}\right)^{2}}{\sum_{i=1}^{g} \sum_{j=1}^{n_{i}}\left(r_{i j}-\bar{r}\right)^{2}}, \text { where: }
$$
- $N$ is the total number of observations across all groups
- $g$ is the number of groups
- $n_{i}$ is the number of observations in group $i$
- $r_{i j}$ is the rank (among all observations) of observation $j$ from group $i$
- $\bar{r}_{i} .=\frac{\sum_{j=1}^{n_{i}} r_{i j}}{n_{i}}$ is the average rank of all observations in group $i$
- $\bar{r}=\frac{1}{2}(N+1)$ is the average of all the $r_{i j}$.
"""

df_satellite = sub_df[sub_df['g']=="Satellite"]
df_GCWAN = sub_df[sub_df['g']=="GC WAN"]
result1 = stats.kruskal(df_satellite['x'],df_GCWAN['x'])
print(result1)
print(df_satellite)

df_Middleware = sub_df[sub_df['g']=="Middleware"]
df_GCWAN = sub_df[sub_df['g']=="GC WAN"]
result2 = stats.kruskal(df_Middleware['x'],df_GCWAN['x'])
print(result2)

# 'Classified Infrastructure', 'Cloud Brokering',
#        'Conferencing Services', 'Contact Centre',
#        'Data Centre Facilities Management', 'Database',
#        'Directory Services', 'Distributed Print', 'Email', 'Firewall',
#        'Fixed Line', 'GC WAN',
#        'Government of Canada Managed Security Service',
#        'High-performance Computing',
#        'Internal Credential Management (ICM)',
#        'Intra-building Network Services', 'Mainframe',
#        'Managed Secure File Transfer', 'Middleware', 'Midrange',
#        'Mobile Devices', 'Other Activities', 'Satellite',
#        'Secure Remote Access', 'Storage', 'Toll-free Voice',
#        'WTD Provisioning', 'Workplace Technology Services'
df_ci = sub_df[sub_df['g']=="Classified Infrastructure"]
df_cs = sub_df[sub_df['g']=="Conferencing Services"]
df_cc = sub_df[sub_df['g']=="Contact Centre"]
df_dcfm = sub_df[sub_df['g']=="Data Centre Facilities Management"]
df_db = sub_df[sub_df['g']=="Database"]
df_ds = sub_df[sub_df['g']=="Directory Services"]
df_dp = sub_df[sub_df['g']=="Distributed Print"]
df_em = sub_df[sub_df['g']=="Email"]
df_f = sub_df[sub_df['g']=="Firewall"]
df_fl = sub_df[sub_df['g']=="Fixed Line"]
df_gw = sub_df[sub_df['g']=="GC WAN"]
df_gcmss = sub_df[sub_df['g']=="Government of Canada Managed Security Service"]
df_hpc = sub_df[sub_df['g']=="High-performance Computing"]
df_icm = sub_df[sub_df['g']=="Internal Credential Management (ICM)"]
df_ins = sub_df[sub_df['g']=="Intra-building Network Services"]
df_mf = sub_df[sub_df['g']=="Mainframe"]
df_msft = sub_df[sub_df['g']=="Managed Secure File Transfer"]
df_mw = sub_df[sub_df['g']=="Middleware"]
df_Midrange = sub_df[sub_df['g']=="Midrange"]
df_mob = sub_df[sub_df['g']=="Mobile Devices"]
df_oa = sub_df[sub_df['g']=="Other Activities"]
df_Satellite = sub_df[sub_df['g']=="Satellite"]
df_sra = sub_df[sub_df['g']=="Secure Remote Access"]
df_Storage = sub_df[sub_df['g']=="Storage"]
df_tv = sub_df[sub_df['g']=="Toll-free Voice"]
df_wtd = sub_df[sub_df['g']=="WTD Provisioning"]
df_wts = sub_df[sub_df['g']=="Workplace Technology Services"]
df_cb = sub_df[sub_df['g']=="Cloud Brokering"]
result_final = stats.kruskal(df_ci['x'],df_cs['x'],df_cb['x'],df_cc['x'],df_dcfm['x'],df_db['x'],df_ds['x'],
              df_dp['x'],df_em['x'],df_f['x'],df_fl['x'],df_gw['x'],df_gcmss['x'],df_hpc['x'],
              df_icm['x'],df_ins['x'],df_mf['x'],df_msft['x'],df_mw['x'],df_Midrange['x'],df_mob['x'],
              df_oa['x'],df_Satellite['x'],df_sra['x'],df_Storage['x'],df_tv['x'],df_wtd['x'],df_wts['x'])
print(result_final)
