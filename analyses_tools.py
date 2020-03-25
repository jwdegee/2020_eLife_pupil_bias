import os, glob
import numpy as np
import scipy as sp
import scipy.stats as stats
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
from joblib import Parallel, delayed

from IPython import embed as shell

# import jw_tools.myfuncs as myfuncs

from tools_mcginley import utils

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
sns.set(style='ticks', font='Arial', font_scale=1, rc={
    'axes.labelsize': 7,
    'axes.titlesize': 7,
    'xtick.labelsize': 6,
    'ytick.labelsize': 6,
    'legend.fontsize': 6,
    'axes.linewidth': 0.25,
    'xtick.major.width': 0.25,
    'ytick.major.width': 0.25,
    'ytick.major.width': 0.25,
    'ytick.major.width': 0.25,
    'ytick.major.pad' : 2.0,
    'ytick.minor.pad' : 2.0,
    'xtick.major.pad' : 2.0,
    'xtick.minor.pad' : 2.0,
    'axes.labelpad' : 4.0,
    'axes.titlepad' : 6.0,
    } )
sns.plotting_context()

def is_outlier(points, thresh=3.5):
    """
    Returns a boolean array with True if points are outliers and False 
    otherwise.

    Parameters:
    -----------
        points : An numobservations by numdimensions array of observations
        thresh : The modified z-score to use as a threshold. Observations with
            a modified z-score (based on the median absolute deviation) greater
            than this value will be classified as outliers.

    Returns:
    --------
        mask : A numobservations-length boolean array.

    References:
    ----------
        Boris Iglewicz and David Hoaglin (1993), "Volume 16: How to Detect and
        Handle Outliers", The ASQC Basic References in Quality Control:
        Statistical Techniques, Edward F. Mykytka, Ph.D., Editor. 
    """
    if len(points.shape) == 1:
        points = points[:,None]
    median = np.median(points, axis=0)
    diff = np.sum((points - median)**2, axis=-1)
    diff = np.sqrt(diff)
    med_abs_deviation = np.median(diff)

    modified_z_score = 0.6745 * diff / med_abs_deviation

    return modified_z_score > thresh

def histogram(df, span):
    fig = plt.figure(figsize=(2,2))
    plt.hist(df['rt'], bins=25)
    if span is not None:
        plt.axvspan(span[0], span[1], color='black', alpha=0.1)
    sns.despine(offset=2, trim=True)
    plt.tight_layout()
    return fig

def plot_responses(df, epochs, epochs_s, span=None):

    fig = plt.figure(figsize=(2,2))
    for e, ls in zip([epochs, epochs_s], ['-', '--']):
        x = np.array(e.columns, dtype=float)
        mean = e.groupby(df['subj_idx']).mean().mean(axis=0)
        sem = e.groupby(df['subj_idx']).mean().sem(axis=0)
        plt.fill_between(x, mean-sem, mean+sem, alpha=0.2, color='black')
        plt.plot(x, mean, color='black', ls=ls)
    if span is not None:
        plt.axvspan(span[0], span[1], color='black', alpha=0.1)
    plt.axvline(0, lw=0.5, color='black')
    plt.xlabel('Time (s)')
    plt.ylabel('Response (% change)')
    sns.despine(offset=2, trim=True)
    plt.tight_layout()
    return fig

def behavior(df, measure='d'):
    """
    Computes d' and criterion
    """
    
    import numpy as np
    import scipy as sp

    # sdt:    
    df['hit'] = (df['response']==1)&(df['correct']==1)
    df['fa'] = (df['response']==1)&(df['correct']==0)
    df['miss'] = (df['response']==0)&(df['correct']==0)
    df['cr'] = (df['response']==0)&(df['correct']==1)

    # rates:
    hit_rate = np.sum(df['hit']) / (np.sum(df['hit']) + np.sum(df['miss']))
    fa_rate = np.sum(df['fa']) / (np.sum(df['fa']) + np.sum(df['cr']))
    if hit_rate > 0.999:
        hit_rate = 0.999
    elif hit_rate < 0.001:
        hit_rate = 0.001
    if fa_rate > 0.999:
        fa_rate = 0.999
    elif fa_rate < 0.001:
        fa_rate = 0.001
    hit_rate_z = stats.norm.isf(1-hit_rate)
    fa_rate_z = stats.norm.isf(1-fa_rate)
    
    if measure == 'rt':
        return df['rt'].mean()
    elif measure == 'd':
        return hit_rate_z - fa_rate_z
    elif measure == 'c':
        return -(hit_rate_z + fa_rate_z) / 2.0

def compute_behavior(df, groupby = ['subj_idx', 'bin']):

    params = pd.DataFrame({'subj_idx': np.array(df.groupby(groupby).first().index.get_level_values('subj_idx')),
                    'bin': np.array(df.groupby(groupby).first().index.get_level_values('bin')),
                    'rt': np.array(df.groupby(groupby).apply(behavior, 'rt')),
                    'd': np.array(df.groupby(groupby).apply(behavior, 'd')),
                    'c': np.array(df.groupby(groupby).apply(behavior, 'c'))})
    
    # add sign-flipped bias:
    params_overall = pd.DataFrame({'subj_idx': np.array(df.groupby(['subj_idx']).first().index.get_level_values('subj_idx')),
                    'rt': np.array(df.groupby(['subj_idx']).apply(behavior, 'rt')),
                    'd': np.array(df.groupby(['subj_idx']).apply(behavior, 'd')),
                    'c': np.array(df.groupby(['subj_idx']).apply(behavior, 'c'))})
    params['cf'] = params['c']
    for subj in params['subj_idx'].unique():
        if params_overall.loc[params_overall['subj_idx'] == subj, 'c'].values < 0:
            params.loc[params['subj_idx'] == subj, 'cf'] = params.loc[params['subj_idx'] == subj, 'cf'] * -1

    return params.melt(id_vars=groupby)

def mixed_linear_modeling(df, x='bin', df_sim=None):

    fig = plt.figure(figsize=(1.25*len(df['variable'].unique()), 1.5))
    plt_nr = 1
    
    for param in df['variable'].unique():
        
        data = df.loc[df['variable']==param,:]

        ax = fig.add_subplot(1,len(df['variable'].unique()), plt_nr)

        # sns.barplot(x='variable', y='value', hue='bin', units='subj_idx', palette='Reds', ci=None, data=df)
        # sns.barplot(x='variable', y='value', hue='bin', units='subj_idx', palette='Reds', ci=66, data=df)
        sns.pointplot(x='bin', y='value', units='subj_idx', join=False, ci=66, scale=0.66, errwidth=1, palette='Reds', data=data, zorder=1, **{'linewidths':0})
        # sns.stripplot(x='variable', y='value', hue='bin', color='grey', size=2, jitter=False, dodge=True, data=df)
        # locs = np.sort(np.array([p.get_x() + p.get_width() / 2. for p in ax.patches]))
        
        # variables:
        data['intercept'] = 1
        data.loc[:,'{}_^2'.format(x)] = np.array(data.loc[:,x]**2)

        # zscore:
        for subj in data['subj_idx'].unique():
            ind = data['subj_idx']==subj
            data.loc[ind,x] = (data.loc[ind,x] - data.loc[ind,x].mean()) / data.loc[ind,x].std()
            data.loc[ind,'{}_^2'.format(x)] = (data.loc[ind,'{}_^2'.format(x)]  - data.loc[ind,'{}_^2'.format(x)].mean()) / data.loc[ind,'{}_^2'.format(x)].std()
        
        endog = data.loc[:,'value'].astype(float)
        exog0 = data.loc[:,['intercept']].astype(float)
        exog1 = data.loc[:,['intercept', x]].astype(float)
        exog2 = data.loc[:,['intercept', x, '{}_^2'.format(x)]].astype(float)

        # comparison:
        try:
            md1 = sm.MixedLM(endog, exog1, data.loc[:,'subj_idx'], exog_re=exog1)
            mdf1 = md1.fit(reml=False)
            mdf1.summary()
        except:
            md1 = sm.MixedLM(endog, exog1, data.loc[:,'subj_idx'])
            mdf1 = md1.fit(reml=False)
            mdf1.summary()
        try:
            md2 = sm.MixedLM(endog, exog2, data.loc[:,'subj_idx'], exog_re=exog2)
            mdf2 = md2.fit(reml=False)
            mdf2.summary()
        except:
            md2 = sm.MixedLM(endog, exog2, data.loc[:,'subj_idx'])
            mdf2 = md2.fit(reml=False)
            mdf2.summary()
        if (mdf1.aic - mdf2.aic) > 10:
            exog = exog2.copy()
        else:
            exog = exog1.copy()

        # refit with reml:
        try:
            md = sm.MixedLM(endog, exog, groups=data.loc[:,'subj_idx'], exog_re=exog)
            mdf = md.fit()
            mdf.summary()
        except:
            md = sm.MixedLM(endog, exog, groups=data.loc[:,'subj_idx'])
            mdf = md.fit()
            mdf.summary()
        print(mdf.summary())
        xx = np.sort(np.array([p.get_data()[0][0] for p in ax.lines]))
        yy = mdf.params['intercept']+(np.array(exog.groupby('bin').mean().index)*mdf.params['bin'])
        plt.plot(xx, yy, lw=2, color='black')
        plt.text(x=xx.min(), y=ax.get_ylim()[0]+((ax.get_ylim()[1]-ax.get_ylim()[0])*0.95), s='p={}'.format(round(mdf.pvalues[x],3)), size=6)

        if not df_sim is None:
            sns.pointplot(x='bin', y='value', color='blue', join=False, ci=None, markers='x', scale=0.66,
                            data=df_sim.loc[df['variable']==param,:].groupby(['variable', 'bin']).mean().reset_index(), zorder=100)

        try:
            plt.gca().get_legend().remove()
        except:
            pass
        plt_nr += 1

        plt.xticks(np.array(ax.get_xticks(), dtype=int))
        plt.ylabel(param)

    sns.despine(offset=2, trim=True)
    plt.tight_layout()
    return fig