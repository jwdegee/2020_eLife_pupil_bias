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
from statsmodels.stats.anova import AnovaRM
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

def cluster_sig_bar_1samp(array, x, yloc, color, ax, threshold=0.05, nrand=5000, cluster_correct=True, n_jobs=10):
    
    import mne

    if cluster_correct:
        whatever, clusters, pvals, bla = mne.stats.permutation_cluster_1samp_test(array, n_permutations=nrand, n_jobs=n_jobs)
        for j, cl in enumerate(clusters):
            if len(cl) == 0:
                pass
            else:
                if pvals[j] < threshold:
                    for c in cl:
                        sig_bool_indices = np.arange(len(x))[c]
                        xx = np.array(x[sig_bool_indices])
                        try:
                            xx[0] = xx[0] - (np.diff(x)[0] / 2.0)
                            xx[1] = xx[1] + (np.diff(x)[0] / 2.0)
                        except:
                            xx = np.array([xx - (np.diff(x)[0] / 2.0), xx + (np.diff(x)[0] / 2.0),]).ravel()
                        # ax.hlines(((ax.get_ylim()[1] - ax.get_ylim()[0]) / yloc)+ax.get_ylim()[0], xx[0], xx[-1], color=color, alpha=1, linewidth=2.5)
                        ax.hlines(yloc, xx[0], xx[-1], color=color, alpha=1, linewidth=2.5)

    else:
        p = np.zeros(array.shape[1])
        for i in range(array.shape[1]):
            # p[i] = sp.stats.ttest_rel(array[:,i], np.zeros(array.shape[0]))[1]
            p[i] = sp.stats.wilcoxon(array[:,i], np.zeros(array.shape[0]))[1]
        sig_indices = np.array(p < 0.05, dtype=int)
        sig_indices[0] = 0
        sig_indices[-1] = 0
        s_bar = zip(np.where(np.diff(sig_indices)==1)[0]+1, np.where(np.diff(sig_indices)==-1)[0])
        for sig in s_bar:
            ax.hlines(((ax.get_ylim()[1] - ax.get_ylim()[0]) / yloc)+ax.get_ylim()[0], x[int(sig[0])]-(np.diff(x)[0] / 2.0), x[int(sig[1])]+(np.diff(x)[0] / 2.0), color=color, alpha=1, linewidth=2.5)
        
    

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

def plot_responses(df, epochs, epochs_s, span=None, stat_by=['subject'], bin_by='pupil_stim_1s'):

    fig = plt.figure(figsize=(3,1.5))
    ax = fig.add_subplot(121)
    for e, ls in zip([epochs, epochs_s], ['-', '--']):
        x = np.array(e.columns, dtype=float)
        mean = e.groupby(df['subject']).mean().mean(axis=0)
        sem = e.groupby(df['subject']).mean().sem(axis=0)
        plt.fill_between(x, mean-sem, mean+sem, alpha=0.2, color='black')
        plt.plot(x, mean, color='black', ls=ls)    
    
    e_for_stat = e.copy()
    x = np.array(e_for_stat.columns, dtype=float)[1:]
    for s in stat_by:
        e_for_stat[s] = df[s]
    e_for_stat = e_for_stat.set_index(stat_by)

    cluster_sig_bar_1samp( np.array(e_for_stat.groupby(stat_by).mean())[:,1:], x, 1, 'black', ax, threshold=0.05, nrand=5000, cluster_correct=True, n_jobs=10)
    if span is not None:
        plt.axvspan(span[0], span[1], color='black', alpha=0.1)
    plt.axvline(0, lw=0.5, color='black')
    plt.axhline(0, lw=0.5, color='black')
    plt.xlabel('Time (s)')
    plt.ylabel('Response (% change)')
    
    
    ax = fig.add_subplot(121)

    shell()
    
    sns.despine(offset=2, trim=True)
    plt.tight_layout()
    return fig

def composite_bias(df):
    
    df['hit'] = ((df['stimulus']==1)&(df['response']==1)).astype(int)
    df['fa'] = ((df['stimulus']==0)&(df['response']==1)).astype(int)
    df['miss'] = ((df['stimulus']==1)&(df['response']==0)).astype(int)
    df['cr'] = ((df['stimulus']==0)&(df['response']==0)).astype(int)

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
                    'rt': np.array(df.groupby(groupby).apply(behavior, 'rt')),
                    'd': np.array(df.groupby(groupby).apply(behavior, 'd')),
                    'c': np.array(df.groupby(groupby).apply(behavior, 'c'))})
    for var in groupby:
        params[var] = np.array(df.groupby(groupby).first().index.get_level_values(var))

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

def mixed_linear_modeling(df, x='bin', bic_diff=10, df_sims=None, colors=None):

    fig = plt.figure(figsize=(1.1*len(df['variable'].unique()), 1.5))
    plt_nr = 1
    
    for param in df['variable'].unique():
        
        data = df.loc[df['variable']==param,:]

        ax = fig.add_subplot(1,len(df['variable'].unique()), plt_nr)

        # sns.barplot(x='variable', y='value', hue='bin', units='subj_idx', palette='Reds', ci=None, data=df)
        # sns.barplot(x='variable', y='value', hue='bin', units='subj_idx', palette='Reds', ci=66, data=df)
        kwargs = {'linewidths':0, 'markeredgewidth':0.5, 'markeredgecolor':'black', 'ecolor':'black'}
        if ('level' in data.columns) & ~(x=='level'):
            sns.pointplot(x=x, y='value', hue='level', units='subj_idx', join=False, ci=66, scale=0.66, errwidth=1, palette='Greys', data=data, zorder=1, **kwargs)
        else:
            sns.pointplot(x=x, y='value', units='subj_idx', join=False, ci=66, scale=0.66, errwidth=1, color='grey', data=data, zorder=1, **kwargs)
        # sns.stripplot(x='variable', y='value', hue='bin', color='grey', size=2, jitter=False, dodge=True, data=df)
        # locs = np.sort(np.array([p.get_x() + p.get_width() / 2. for p in ax.patches]))

        if param == 'rt':
            mean_rt = data['value'].mean()
            plt.ylim(mean_rt-0.25, mean_rt+0.25)
        
        if len(data[x].unique()) > 2:
            # variables:
            data['intercept'] = 1
            data.loc[:,'{}_^2'.format(x)] = np.array(data.loc[:,x]**2)

            # # zscore:
            # for subj in data['subj_idx'].unique():
            #     ind = data['subj_idx']==subj
            #     data.loc[ind,x] = (data.loc[ind,x] - data.loc[ind,x].mean()) / data.loc[ind,x].std()
            #     data.loc[ind,'{}_^2'.format(x)] = (data.loc[ind,'{}_^2'.format(x)]  - data.loc[ind,'{}_^2'.format(x)].mean()) / data.loc[ind,'{}_^2'.format(x)].std()
            
            endog = data.loc[:,'value'].astype(float)
            if ('level' in data.columns) & ~(x=='level'):
                exog1 = data.loc[:,['intercept', 'level', x]].astype(float)
                exog2 = data.loc[:,['intercept', 'level', x, '{}_^2'.format(x)]].astype(float)
            else:
                exog1 = data.loc[:,['intercept', x]].astype(float)
                exog2 = data.loc[:,['intercept', x, '{}_^2'.format(x)]].astype(float)


            # comparison:
            md1 = sm.MixedLM(endog, exog1, data.loc[:,'subj_idx'], exog_re=exog1)
            mdf1 = md1.fit(reml=False)
            md2 = sm.MixedLM(endog, exog2, data.loc[:,'subj_idx'], exog_re=exog2)
            mdf2 = md2.fit(reml=False)
            if mdf1.converged & mdf2.converged:
                random = True
            else:
                md1 = sm.MixedLM(endog, exog1, data.loc[:,'subj_idx'],)
                mdf1 = md1.fit(reml=False)
                md2 = sm.MixedLM(endog, exog2, data.loc[:,'subj_idx'],)
                mdf2 = md2.fit(reml=False)
                random = False
            if (mdf1.bic - mdf2.bic) > bic_diff:
                exog = exog2.copy()
            else:
                exog = exog1.copy()

            # refit with reml:
            if random:
                mdf = sm.MixedLM(endog, exog, groups=data.loc[:,'subj_idx'], exog_re=exog).fit()
            else:
                mdf = sm.MixedLM(endog, exog, groups=data.loc[:,'subj_idx']).fit()
            print(mdf.summary())
            xx = np.sort(np.array([p.get_data()[0][0] for p in ax.lines]))
            if ('level' in data.columns) & ~(x=='level'):
                if (mdf1.bic - mdf2.bic) > bic_diff:
                    yy = np.concatenate([mdf.params['intercept']+(np.array(exog.groupby('level').mean().index)*mdf.params['level']) + 
                                                (b*mdf.params[x]) + ((b**2)*mdf.params['{}_^2'.format(x)]) for b in np.array(exog.groupby(x).mean().index)])
                    plt.title('p = {}\np1 = {}\np2 = {}'.format(round(mdf.pvalues['level'],3), round(mdf.pvalues[x],3), round(mdf.pvalues['{}_^2'.format(x)],3)), size=6)
                else:
                    yy = np.concatenate([mdf.params['intercept']+(np.array(exog.groupby('level').mean().index)*mdf.params['level']) + 
                                                (b*mdf.params[x]) for b in np.array(exog.groupby(x).mean().index)])
                    plt.title('p = {}\np = {}'.format(round(mdf.pvalues['level'],3), round(mdf.pvalues[x],3)), size=6)
                for v in exog.groupby('level').mean().index:
                    plt.plot(xx[int(v)::len(exog.groupby('level').mean().index)], yy[int(v)::len(exog.groupby('level').mean().index)], lw=1, color='black')
            else:
                if (mdf1.bic - mdf2.bic) > bic_diff:
                    yy = mdf.params['intercept']+(np.array(exog.groupby(x).mean().index)*mdf.params[x])+((np.array(exog.groupby(x).mean().index)**2)*mdf.params['{}_^2'.format(x)])
                    plt.title('p1 = {}\np2 = {}'.format(round(mdf.pvalues[x],3),round(mdf.pvalues['{}_^2'.format(x)],3)), size=6)
                else:    
                    yy = mdf.params['intercept']+(np.array(exog.groupby(x).mean().index)*mdf.params[x])
                    plt.title('p = {}'.format(round(mdf.pvalues[x],3)), size=6)
                plt.plot(xx, yy, lw=1, color='black')

        if not df_sims is None:
            if ('level' in data.columns) & ~(x=='level'):
                for df_sim, color in zip(df_sims, colors):
                    sns.pointplot(x=x, y='value', hue='level', palette=['blue' for _ in range(len(data['level'].unique()))], join=False, ci=None, markers='x', scale=0.66,
                    data=df_sim.loc[df['variable']==param,:], zorder=100)
            else:
                for df_sim, color in zip(df_sims, colors):
                    sns.pointplot(x=x, y='value', color='blue', join=False, ci=None, markers='x', scale=0.66,
                            data=df_sim.loc[df['variable']==param,:], zorder=100)
        try:
            plt.gca().get_legend().remove()
        except:
            pass
        
        plt.xticks(ax.get_xticks(), list(np.array(ax.get_xticks(), dtype=int)))
        plt.ylabel(param)
        
        plt_nr += 1

    sns.despine(offset=2, trim=True)
    plt.tight_layout()
    return fig

def conditional_response_plot(df, quantiles=[0,0.1,0.3,0.5,0.7,0.9,1], y='response', ylim=None, df_sims=None, color=None):
    
    fig = plt.figure(figsize=(1.5,1.5))
    ax = fig.add_subplot(1,1,1)
    
    df['rt_bin'] = df.groupby(['subj_idx', 'bin'])['rt'].apply(pd.qcut, quantiles, labels=False)
    d = df.groupby(['subj_idx', 'bin', 'rt_bin']).mean().reset_index().groupby(['bin', 'rt_bin']).mean().reset_index()
    ds = df.groupby(['subj_idx', 'bin', 'rt_bin']).mean().reset_index().groupby(['bin', 'rt_bin']).sem().reset_index()
    plt.errorbar(x=d.loc[d['bin']==min(d['bin']), 'rt'], y=d.loc[d['bin']==min(d['bin']), y], yerr=ds.loc[ds['bin']==min(ds['bin']), y], color='black', ls='--')
    plt.errorbar(x=d.loc[d['bin']==max(d['bin']), 'rt'], y=d.loc[d['bin']==max(d['bin']), y], yerr=ds.loc[ds['bin']==max(ds['bin']), y], color='black', ls='-')

    aovrm = AnovaRM(df.groupby(['subj_idx', 'bin', 'rt_bin']).mean().reset_index(), y, 'subj_idx', within=['bin', 'rt_bin'], aggregate_func='mean')
    res = aovrm.fit()
    plt.title('bin: p={}; int: p={}'.format(round(res.anova_table.iloc[0]['Pr > F'],3),round(res.anova_table.iloc[2]['Pr > F'],3) ))

    if not df_sims is None:
        for df_sim, color in zip(df_sims, colors):
            df_sim['rt_bin'] = df_sim.groupby(['subj_idx', 'bin'])['rt'].apply(pd.qcut, quantiles, labels=False)
            d = df_sim.groupby(['subj_idx', 'bin', 'rt_bin']).mean().reset_index().groupby(['bin', 'rt_bin']).mean().reset_index()
            plt.scatter(x=d.loc[d['bin']==min(d['bin']), 'rt'], y=d.loc[d['bin']==min(d['bin']), y], marker='x', color=color)
            plt.scatter(x=d.loc[d['bin']==max(d['bin']), 'rt'], y=d.loc[d['bin']==max(d['bin']), y], marker='x', color=color)

    # ax.set_title('P(corr.)={}, {}, {}\nP(bias)={}, {}, {}'.format(*means))
    plt.axhline(0.5, lw=0.5, color='k')
    if ylim is not None:
        ax.set_ylim(ylim)
    ax.set_xlabel('RT (s)')
    ax.set_ylabel('P({})'.format(y))
    sns.despine(trim=True, offset=2)
    plt.tight_layout()

    return fig