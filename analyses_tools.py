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
        Boris Iglewicz and David Hoaglin (1993), "level 16: How to Detect and
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
        mean = e.groupby(df['subject']).mean().mean(axis=0)
        sem = e.groupby(df['subject']).mean().sem(axis=0)
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

def composite_bias(df):
    
    # rates:
    hit_rates = np.array([np.sum(d['hit']) / (np.sum(d['hit']) + np.sum(d['miss'])) for v,d in df.groupby(['level'])])
    fa_rate = np.array([np.sum(d['fa']) / (np.sum(d['fa']) + np.sum(d['cr'])) for v,d in df.groupby(['level'])])[0]

    hit_rates[hit_rates>0.999] = 0.999
    if fa_rate<0.001:
        fa_rate = 0.001

    hit_rates_z = sp.stats.norm.isf(1-hit_rates)
    fa_rate_z = sp.stats.norm.isf(1-fa_rate)
    d_primes = hit_rates_z - fa_rate_z
    criterions = -(hit_rates_z + fa_rate_z)/2

    choice_points = (0.5*d_primes)+criterions

    x = np.linspace(0, max(d_primes), 10000)
    noise_dist = sp.stats.norm.pdf(x,0,1,)
    signal_dist = np.zeros(len(x))
    for d_prime in d_primes:
        signal_dist = signal_dist + sp.stats.norm.pdf(x,d_prime,1,)
    signal_dist = signal_dist / len(d_primes)
    diff_dist = abs(noise_dist-signal_dist)
    
    choice_point_neutral = x[np.where(diff_dist==min(diff_dist))[0][0]]

    bias = choice_points - choice_point_neutral

    return bias[0]

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

def prepare_df(df):
    dfs = []
    for subj in np.unique(df['subj_idx']):
        df_subj = df.loc[df['subj_idx']==subj,:]
        df_subj_noise = df_subj.loc[df_subj['stimulus']==0,:]
        df_subj_stim = df_subj.loc[df_subj['stimulus']==1,:]
        for vol in np.unique(df_subj_stim['level']):
            df_subj_noise.loc[:, 'level'] = vol
            df_subj_stim_vol = df_subj_stim.loc[df_subj_stim['level']==vol,:]
            dfs.append(pd.concat((df_subj_noise, df_subj_stim_vol), axis=0))
    df = pd.concat(dfs, axis=0).reset_index(drop=True)
    df.loc[:, 'level'] = df.loc[:, 'level'].astype(int)
    return df

def compute_behavior(df, groupby=['subj_idx', 'bin']):

    params = pd.DataFrame({'subj_idx': np.array(df.groupby(groupby).first().index.get_level_values('subj_idx')),
                    'bin': np.array(df.groupby(groupby).first().index.get_level_values('bin')),
                    'rt': np.array(df.groupby(groupby).apply(behavior, 'rt')),
                    'd': np.array(df.groupby(groupby).apply(behavior, 'd')),
                    'c': np.array(df.groupby(groupby).apply(behavior, 'c'))})
    if 'level' in df.columns:
        params['level'] = np.array(df.groupby(groupby).first().index.get_level_values('level'))

    # add sign-flipped bias:
    params_overall = pd.DataFrame({'subj_idx': np.array(df.groupby(['subj_idx']).first().index.get_level_values('subj_idx')),
                    'rt': np.array(df.groupby(['subj_idx']).apply(behavior, 'rt')),
                    'd': np.array(df.groupby(['subj_idx']).apply(behavior, 'd')),
                    'c': np.array(df.groupby(['subj_idx']).apply(behavior, 'c'))})
    params['cf'] = params['c']
    for subj in params['subj_idx'].unique():
        if params_overall.loc[params_overall['subj_idx'] == subj, 'c'].values < 0:
            params.loc[params['subj_idx'] == subj, 'cf'] = params.loc[params['subj_idx'] == subj, 'cf'] * -1

    return params

def mixed_linear_modeling(df, x='bin', df_sim=None, bic_diff=5):

    fig = plt.figure(figsize=(1.25*len(df['variable'].unique()), 1.5))
    plt_nr = 1
    
    for param in df['variable'].unique():
        
        data = df.loc[df['variable']==param,:]

        ax = fig.add_subplot(1,len(df['variable'].unique()), plt_nr)

        # sns.barplot(x='variable', y='value', hue='bin', units='subj_idx', palette='Reds', ci=None, data=df)
        # sns.barplot(x='variable', y='value', hue='bin', units='subj_idx', palette='Reds', ci=66, data=df)
        if 'level' in data.columns:
            sns.pointplot(x='bin', y='value', hue='level', units='subj_idx', join=False, ci=66, scale=0.66, errwidth=1, palette='Reds', data=data, zorder=1, **{'linewidths':0})
        else:
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
        if 'level' in data.columns:
            exog1 = data.loc[:,['intercept', 'level', x]].astype(float)
            exog2 = data.loc[:,['intercept', 'level', x, '{}_^2'.format(x)]].astype(float)
        else:
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
        if (mdf1.aic - mdf2.aic) > bic_diff:
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
        if 'level' in data.columns:
            if (mdf1.aic - mdf2.aic) > bic_diff:
                yy = np.concatenate([mdf.params['intercept']+(np.array(exog.groupby('level').mean().index)*mdf.params['level']) + 
                                            (b*mdf.params[x]) + ((b**2)*mdf.params['{}_^2'.format(x)]) for b in np.array(exog.groupby('bin').mean().index)])
            else:
                yy = np.concatenate([mdf.params['intercept']+(np.array(exog.groupby('level').mean().index)*mdf.params['level']) + 
                                            (b*mdf.params[x]) for b in np.array(exog.groupby('bin').mean().index)])
            for v in exog.groupby('level').mean().index:
                plt.plot(xx[int(v)::len(exog.groupby('level').mean().index)], yy[int(v)::len(exog.groupby('level').mean().index)], lw=1, color='black')
            plt.text(x=xx.min(), y=ax.get_ylim()[0]+((ax.get_ylim()[1]-ax.get_ylim()[0])*0.95), s='p={}; p={}'.format(round(mdf.pvalues[x],3), round(mdf.pvalues['level'],3)), size=6)
        else:
            if (mdf1.aic - mdf2.aic) > bic_diff:
                yy = mdf.params['intercept']+(np.array(exog.groupby('bin').mean().index)*mdf.params[x])+((np.array(exog.groupby('bin').mean().index)**2)*mdf.params['{}_^2'.format(x)])
            else:    
                yy = mdf.params['intercept']+(np.array(exog.groupby('bin').mean().index)*mdf.params[x])
            plt.plot(xx, yy, lw=1, color='black')
            plt.text(x=xx.min(), y=ax.get_ylim()[0]+((ax.get_ylim()[1]-ax.get_ylim()[0])*0.95), s='p={}'.format(round(mdf.pvalues[x],3)), size=6)

        if not df_sim is None:
            if 'level' in data.columns:
                sns.pointplot(x='bin', y='value', color='blue', join=False, ci=None, markers='x', scale=0.66,
                            data=df_sim.loc[df['variable']==param,:].groupby(['variable', 'bin']).mean().reset_index(), zorder=100)
            else:
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