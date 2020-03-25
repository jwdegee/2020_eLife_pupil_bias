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

from tools_mcginley import utils
from analyses_tools import *

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

# variables:
project_dir = '/home/jwdegee/2020_eLife_pupil'
exp_names = [
    'yesno_audio',
    'bias_manipulation_30', 
    'bias_manipulation_70',
    'image_recognition',
    ]
rt_cutoffs = [
    (0.55, 2.5),
    (0.55, 2.5), 
    (0.55, 2.5),
    (0.55, 7.5),
    ]

pupil_cutoffs = [
    [(0,0.5), (-0.55, -0.05)],
    [(0,0.5), (-0.55, -0.05)],
    [(0,0.5), (-0.55, -0.05)],
    [(0,0.5), (-0.55, -0.05)],
    ]

nrs_bins = [
    5,
    3,
    3,
    2,
    ]

# for analyse_exp in [0,1,2,3]:
for analyse_exp in [2]:

    exp_name, rt_cutoff, pupil_cutoff, nr_bin = exp_names[analyse_exp], rt_cutoffs[analyse_exp], pupil_cutoffs[analyse_exp], nrs_bins[analyse_exp]

    # load data:
    df = pd.read_csv(os.path.join(project_dir, 'data', 'df_meta_{}.hdf'.format(exp_name)))
    epoch_p_stim = pd.read_hdf(os.path.join(project_dir, 'data', 'epoch_p_stim_{}.hdf'.format(exp_name)))
    epoch_p_s_stim = pd.read_hdf(os.path.join(project_dir, 'data', 'epoch_p_s_stim_{}.hdf'.format(exp_name)))
    epoch_b_stim = pd.read_hdf(os.path.join(project_dir, 'data', 'epoch_b_stim_{}.hdf'.format(exp_name)))
    try:
        epoch_p_resp = pd.read_hdf(os.path.join(project_dir, 'data', 'epoch_p_resp_{}.hdf'.format(exp_name)))
        epoch_p_s_resp = pd.read_hdf(os.path.join(project_dir, 'data', 'epoch_p_s_resp_{}.hdf'.format(exp_name)))
        epoch_b_resp = pd.read_hdf(os.path.join(project_dir, 'data', 'epoch_b_resp_{}.hdf'.format(exp_name)))
    except:
        pass
    
    # variables:
    df['hit'] = ((df['stimulus']==1)&(df['response']==1)).astype(int)
    df['fa'] = ((df['stimulus']==0)&(df['response']==1)).astype(int)
    df['miss'] = ((df['stimulus']==1)&(df['response']==0)).astype(int)
    df['cr'] = ((df['stimulus']==0)&(df['response']==0)).astype(int)
    
    # rt distribution:
    fig = histogram(df, rt_cutoff)
    fig.savefig(os.path.join(project_dir, 'figs', 'rt_distribution_{}.pdf'.format(exp_name)))

    # timepoints:
    x_stim = np.array(epoch_p_stim.columns, dtype=float)
    x_s_stim = np.array(epoch_p_s_stim.columns, dtype=float)
    x_resp = np.array(epoch_p_resp.columns, dtype=float)
    x_s_resp = np.array(epoch_p_s_resp.columns, dtype=float)
    
    # add blinks:
    df['blink'] = np.array(epoch_b_resp.loc[:,(x_resp>=pupil_cutoff[1][0])&(x_resp<=pupil_cutoff[1][1])].sum(axis=1) > 0.5)

    # add pupil values:
    # df['pupil_stim_1s'] = np.array(epoch_p_s_stim.loc[:,(x_s_stim>pupil_cutoff[0][0])&(x_s_stim<pupil_cutoff[0][1])].max(axis=1))
    # df['pupil_resp_1s'] = np.array(epoch_p_s_resp.loc[:,(x_s_resp>pupil_cutoff[1][0])&(x_s_resp<pupil_cutoff[1][1])].max(axis=1))
    df['pupil_stim_1s'] = np.array(epoch_p_s_stim.loc[:,(x_s_stim>pupil_cutoff[0][0])&(x_s_stim<pupil_cutoff[0][1])].quantile(0.95, axis=1))
    df['pupil_resp_1s'] = np.array(epoch_p_s_resp.loc[:,(x_s_resp>pupil_cutoff[1][0])&(x_s_resp<pupil_cutoff[1][1])].quantile(0.95, axis=1))
    
    # if (exp_name == 'bias_manipulation_30') or (exp_name == 'bias_manipulation_70'):
    #     df['pupil_stim_0'] = epoch_p_stim.loc[:,(x_stim>-0.5)&(x_stim<0)].mean(axis=1)
    #     df['pupil_stim_1'] = epoch_p_stim.loc[:,(x_stim>0)&(x_stim<1)].mean(axis=1) - df['pupil_stim_0']
    # if (exp_name == 'bias_manipulation_30') or (exp_name == 'bias_manipulation_70'):
    #     df['pupil_resp_1'] = epoch_p_resp.loc[:,(x_resp>-1)&(x_resp<1.5)].mean(axis=1) - df['pupil_stim_0']

    # omissions step 1:
    omissions = (
        np.zeros(df.shape[0], dtype=bool)
        + np.array(df['response']==-1)
        + np.array(np.isnan(df['rt']))
        + np.array(df['rt'] < rt_cutoff[0])
        + np.array(df['rt'] > rt_cutoff[1])
        + np.array(df['blink']==1)
        + np.array(np.isnan(df['pupil_stim_1s']))
        + np.array(np.isnan(df['pupil_resp_1s']))
        + is_outlier(np.array(df['pupil_stim_1s']))
        + is_outlier(np.array(df['pupil_resp_1s']))
        )
    if exp_name == 'image_recognition':
        omissions = omissions + np.isnan(df['emotional'])
    
    df = df.loc[~omissions,:].reset_index(drop=True)
    epoch_p_stim = epoch_p_stim.loc[~omissions,:].reset_index(drop=True)
    epoch_p_resp = epoch_p_resp.loc[~omissions,:].reset_index(drop=True)
    epoch_p_s_stim = epoch_p_s_stim.loc[~omissions,:].reset_index(drop=True)
    epoch_p_s_resp = epoch_p_s_resp.loc[~omissions,:].reset_index(drop=True)

    # # correct phasic pupil measures:
    # for (subj, ses), d in df.groupby(['subject', 'session']):
    #     ind = (df['subject']==subj)&(df['session']==ses)&~np.isnan(df['pupil_1'])
    #     df.loc[ind, 'pupil_1'] = myfuncs.lin_regress_resid(df.loc[ind, 'pupil_1'], [df.loc[ind, 'rt'],]) + df.loc[ind, 'pupil_1'].mean()
    #     # df.loc[ind, 'pupil_1s_stim'] = myfuncs.lin_regress_resid(df.loc[ind, 'pupil_1s_stim'], [df.loc[ind, 'pupil_0']]) + df.loc[ind, 'pupil_1s_stim'].mean()
    #     # df.loc[ind, 'pupil_1s_resp'] = myfuncs.lin_regress_resid(df.loc[ind, 'pupil_1s_resp'], [df.loc[ind, 'pupil_0']]) + df.loc[ind, 'pupil_1s_resp'].mean()

    # add more variables:
    subjects = df['subject'].unique()
    df['subj_idx'] = np.concatenate(np.array([np.repeat(i, sum(df['subject'] == subjects[i])) for i in range(len(subjects))]))

    # # initialize behaviour:
    # behavior = myfuncs.behavior(df)

    # # save to file:
    # df.to_csv(os.path.join(project_directory, 'yesno_data.csv'))

    # # save to file for DDM:
    # df = df.loc[:,['subj_idx', 'correct', 'stimulus', 'choice_a', 'rt', 'pupil_0', 'pupil_1s_stim', 'pupil_1s_resp', 'run', 'session', 'signal_probability']]
    # df.loc[:,'signal_probability'] = (df.loc[:,'signal_probability']*10).astype(int)
    # df['correct'] = df['correct'].astype(int)
    # df['stimulus'] = df['stimulus'].astype(int)
    # df['choice_a'] = df['choice_a'].astype(int)
    # df.loc[df['signal_probability']==3,:].to_csv(os.path.join(project_directory, '2019_human_auditory_yesno_30.csv'))
    # df.loc[df['signal_probability']==7,:].to_csv(os.path.join(project_directory, '2019_human_auditory_yesno_70.csv'))

    # for s in [0.3, 0.7]:
    #     rs = []
    #     for subj_idx in range(nr_subjects):
    #         d = df.query('(subj_idx == {}) & (signal_probability == {})'.format(subj_idx, s))
    #         rs.append(sp.stats.pearsonr(d['pupil_1s_resp'], d['pupil_0'])[0])
    #     print(np.mean(np.array(rs)))
    #     print(sp.stats.sem(np.array(rs)))

    # save for ddm fitting:
    df.to_csv(os.path.join(project_dir, 'data', 'ddm', '{}.csv'.format(exp_name)))
    
    # pupil responses:
    fig = plot_responses(df, epoch_p_stim, epoch_p_s_stim, span=pupil_cutoff[0])
    fig.savefig(os.path.join(project_dir, 'figs', 'responses_stim_{}.pdf'.format(exp_name)))
    fig = plot_responses(df, epoch_p_resp, epoch_p_s_resp, span=pupil_cutoff[1])
    fig.savefig(os.path.join(project_dir, 'figs', 'responses_resp_{}.pdf'.format(exp_name)))

    # sdt bars:
    for bin_measure in ['pupil_resp_1s', 'pupil_stim_1s']:
        df['bin'] = df.groupby(['subj_idx'])[bin_measure].apply(pd.qcut, q=nr_bin, labels=False)
        params = compute_behavior(df=df, groupby=['subj_idx', 'bin'])
        
        # add pupil response:
        params[bin_measure] = np.NaN
        for (v, b), params_cut in params.groupby(['variable', 'bin']):
            if len(params_cut.index) == len(params['subj_idx'].unique()):
                params.loc[params_cut.index, bin_measure] = np.array(df.groupby(['subj_idx', 'bin']).mean().reset_index().query('bin=={}'.format(b))[bin_measure])

        # plot:
        fig = mixed_linear_modeling(params, 'bin')
        # fig = mixed_linear_modeling(params, bin_measure)
        fig.savefig(os.path.join(project_dir, 'figs', 'bars_sdt_{}_{}.pdf'.format(exp_name, bin_measure)))