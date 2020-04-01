import os, itertools
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.stats.anova import AnovaRM
from joblib import Parallel, delayed
from IPython import embed as shell

from accumodels import plot_tools, pyddm_tools
import analyses_tools
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

def params_melt(params, model_settings):

    try:
        params = params.loc[:,params.columns!='Unnamed: 0']
    except:
        pass
    params = params.melt(id_vars=['subj_idx'])
    for i in range(params.shape[0]):
        variable = "".join(itertools.takewhile(str.isalpha, params.loc[i,'variable']))
        if variable in model_settings['depends_on']: 
            conditions = model_settings['depends_on'][variable]
            if conditions is not None:
                if len(conditions) == 2:
                    params.loc[i,conditions[0]] = int(params.loc[i,'variable'][-3])
                    params.loc[i,conditions[1]] = int(params.loc[i,'variable'][-1])
                elif len(conditions) == 1:
                    params.loc[i,conditions[0]] = int(params.loc[i,'variable'][-1])
        params.loc[i,'variable'] = variable
    
    return params


# all model elements:
# -------------------

project_dir = '/home/jwdegee/repos/2020_eLife_pupil_bias/'
experiment_names = ['yesno_audio', 'bias_manipulation_30', 'bias_manipulation_70', 'bias_manipulation', 'image_recognition']
bin_measures = ['pupil_resp_1s', 'pupil_resp_1s', 'pupil_resp_1s', 'pupil_resp_1s', 'pupil_resp_1s']
nrs_bins = [5,3,3,3,2]

n_jobs = 15
analysis_step = 0
versions = [1,2,3]

# for analyse_exp in [0,1,2]:
for analyse_exp in [3]:

    experiment_name, bin_measure, n_bins = experiment_names[analyse_exp], bin_measures[analyse_exp], nrs_bins[analyse_exp]

    # load data:
    if experiment_name == 'bias_manipulation':
        df = pd.concat((pd.read_csv(os.path.join(project_dir, 'data', 'ddm', '{}_30.csv'.format(experiment_name))),
                        pd.read_csv(os.path.join(project_dir, 'data', 'ddm', '{}_70.csv'.format(experiment_name))))).reset_index(drop=True)
        df['cons'] = df['signal_probability'].copy()
        df['cons'] = (df['cons']<50).astype(int)
        df['bin'] = df.groupby(['subj_idx', 'cons'])[bin_measure].apply(pd.qcut, q=n_bins, labels=False)
    elif exp_name == 'image_recognition':
        df['bin'] = df.groupby(['subj_idx', 'emotional'])[bin_measure].apply(pd.qcut, q=n_bins, labels=False)
    else:
        df = pd.read_csv(os.path.join(project_dir, 'data', 'ddm', '{}.csv'.format(experiment_name)))
        df['bin'] = df.groupby(['subj_idx'])[bin_measure].apply(pd.qcut, q=n_bins, labels=False)

    # fix columns:
    df.loc[df['stimulus']==0, 'stimulus'] = -1

    # compute T_dur:
    T_dur = df['rt'].max()+1

    # set options:
    if experiment_name == 'bias_manipulation':
            model_settings = [
            {'urgency':False, 'T_dur':T_dur, 'depends_on': {'a':['bin'], 'u':None,    'v':['bin'], 't':['bin'], 'z':['cons', 'bin'], 'b':['cons', 'bin']}},
            {'urgency':False, 'T_dur':T_dur, 'depends_on': {'a':None,    'u':None,    'v':None,    't':None,    'z':['cons', 'bin'], 'b':['cons']       }},
            {'urgency':False, 'T_dur':T_dur, 'depends_on': {'a':None,    'u':None,    'v':None,    't':None,    'z':['cons'],        'b':['cons', 'bin']}},
            {'urgency':True,  'T_dur':T_dur, 'depends_on': {'a':None,    'u':['bin'], 'v':None,    't':None,    'z':['cons'],        'b':['cons']       }},
            ]
    else:
        model_settings = [
            {'urgency':False, 'T_dur':T_dur, 'depends_on': {'a':['bin'], 'u':None,    'v':['bin'], 't':['bin'], 'z':['bin'], 'b':['bin']}},
            {'urgency':False, 'T_dur':T_dur, 'depends_on': {'a':None,    'u':None,    'v':None,    't':None,    'z':['bin'], 'b':None   }},
            {'urgency':False, 'T_dur':T_dur, 'depends_on': {'a':None,    'u':None,    'v':None,    't':None,    'z':None,    'b':['bin']}},
            {'urgency':True,  'T_dur':T_dur, 'depends_on': {'a':None,    'u':['bin'], 'v':None,    't':None,    'z':None,    'b':None   }},
            ]

    # cut dataframe:
    if experiment_name == 'bias_manipulation':
        df_emp = df.loc[:,['subj_idx', 'response', 'rt', 'stimulus', 'correct', 'cons', 'bin', bin_measure]]
    else:
        df_emp = df.loc[:,['subj_idx', 'response', 'rt', 'stimulus', 'correct', 'bin', bin_measure]]
    
    # fit model:
    if analysis_step == 0:
        for version in versions:
            
            # fit:
            res = Parallel(n_jobs=n_jobs, verbose=1, backend='loky')(delayed(pyddm_tools.fit_model)(data, model_settings[version]) 
                                        for subj_idx, data in df_emp.groupby(['subj_idx']))
            # params:
            model_nr = 0
            params = []
            for subj_idx, data in df_emp.groupby(['subj_idx']):
                param_names = []
                for component in res[model_nr].dependencies: 
                    param_names = param_names + component.required_parameters
                pars = pd.DataFrame(np.atleast_2d([p.real for p in res[model_nr].get_model_parameters()]), columns=param_names)
                pars['bic'] = res[model_nr].fitresult.value() 
                pars['subj_idx'] = subj_idx
                params.append(pars)
                model_nr += 1
            params = pd.concat(params).reset_index(drop=True)
            
            # save:
            params.to_csv(os.path.join(project_dir, 'fits', '{}_{}.csv'.format(experiment_name, version)))

    # simulate data:
    elif analysis_step == 1:
        for version in versions:
            params = pd.read_csv(os.path.join(project_dir, 'fits', '{}_{}.csv'.format(experiment_name, version)))
            df_sim = pd.concat(Parallel(n_jobs=n_jobs, verbose=1, backend='loky')(delayed(pyddm_tools.simulate_data)(data, params, model_settings[version], subj_idx, 100000)
                                for subj_idx, data in df_emp.groupby(['subj_idx']))).reset_index()
            df_sim.to_csv(os.path.join(project_dir, 'fits', '{}_{}_df_sim.csv'.format(experiment_name, version)))

    # analyses:
    elif analysis_step == 2:
        bics = []
        resids = []
        sdt_emps = []
        sdt_sims = []
        for version in versions:

            print(version)

            # load:
            params = pd.read_csv(os.path.join(project_dir, 'fits', '{}_{}.csv'.format(experiment_name, version)))
            params = params_melt(params, model_settings[version])
            df_sim = pd.read_csv(os.path.join(project_dir, 'fits', '{}_{}_df_sim.csv'.format(experiment_name, version)))

            # empirical and simulated SDT:
            if experiment_name == 'bias_manipulation':
                sdt_emp = compute_behavior(df=df_emp, groupby=['subj_idx', 'cons', 'bin']).melt(id_vars=['subj_idx', 'cons', 'bin'])
                sdt_sim = compute_behavior(df=df_sim, groupby=['subj_idx', 'cons', 'bin']).melt(id_vars=['subj_idx', 'cons', 'bin'])
            else:
                sdt_emp = compute_behavior(df=df_emp, groupby=['subj_idx', 'bin']).melt(id_vars=['subj_idx', 'bin'])
                sdt_sim = compute_behavior(df=df_sim, groupby=['subj_idx', 'bin']).melt(id_vars=['subj_idx', 'bin'])
            sdt_emps.append(sdt_emp)
            sdt_sims.append(sdt_sim)
            # # add pupil:
            # params[bin_measure] = np.NaN
            # sdt_emp[bin_measure] = np.NaN
            # sdt_sim[bin_measure] = np.NaN
            # for (v, b), params_cut in params.groupby(['variable', 'bin']):
            #     if len(params_cut.index) == len(params['subj_idx'].unique()):
            #         params.loc[params_cut.index, bin_measure] = np.array(df_emp.groupby(['subj_idx', 'bin']).mean().reset_index().query('bin=={}'.format(b))[bin_measure])
            # for (v, b), params_cut in sdt_emp.groupby(['variable', 'bin']):
            #     if len(params_cut.index) == len(sdt_emp['subj_idx'].unique()):
            #         sdt_emp.loc[params_cut.index, bin_measure] = np.array(df_emp.groupby(['subj_idx', 'bin']).mean().reset_index().query('bin=={}'.format(b))[bin_measure])
            # for (v, b), params_cut in sdt_sim.groupby(['variable', 'bin']):
            #     if len(params_cut.index) == len(sdt_sim['subj_idx'].unique()):
            #         sdt_sim.loc[params_cut.index, bin_measure] = np.array(df_emp.groupby(['subj_idx', 'bin']).mean().reset_index().query('bin=={}'.format(b))[bin_measure])

            # # model fit:
            # if experiment_name == 'bias_manipulation':
            #     for (c,b), d in df_emp.groupby(['cons', 'bin']):
            #         fig = plot_tools.summary_plot_group(df_emp.loc[(df_emp['cons']==c)&(df_emp['bin']==b),:], df_sim.loc[(df_sim['cons']==c)&(df_sim['bin']==b),:])
            #         fig.savefig(os.path.join(project_dir, 'figs', 'ddm', '{}_{}_model_fit_{}_{}.pdf'.format(experiment_name, version, c, b)))
            # else:
            #     for (c,b), d in df_emp.groupby(['bin']):
            #         fig = plot_tools.summary_plot_group(df_emp.loc[df_emp['bin']==b,:], df_sim.loc[df_sim['bin']==b,:])
            #         fig.savefig(os.path.join(project_dir, 'figs', 'ddm', '{}_{}_model_fit_{}.pdf'.format(experiment_name, version, b)))

            # analysis:
            if experiment_name == 'bias_manipulation':
                for cons in [0,1]:
                    fig = mixed_linear_modeling(params.loc[~pd.isnull(params['bin']) & ((params['cons']==cons)|pd.isna(params['cons'])),:], x='bin')
                    # fig = mixed_linear_modeling(params.loc[~pd.isnull(params['bin']),:], x=bin_measure)
                    fig.savefig(os.path.join(project_dir, 'figs', 'ddm', '{}_{}_bars_{}.pdf'.format(experiment_name, version, cons)))
            else:
                fig = mixed_linear_modeling(params.loc[~pd.isnull(params['bin']),:], x='bin')
                # fig = mixed_linear_modeling(params.loc[~pd.isnull(params['bin']),:], x=bin_measure)
                fig.savefig(os.path.join(project_dir, 'figs', 'ddm', '{}_{}_bars.pdf'.format(experiment_name, version)))

            # resid:
            if experiment_name == 'bias_manipulation':
                resid = pd.DataFrame(sdt_emp.loc[sdt_emp['variable']=='c', ['subj_idx', 'cons', 'bin', 'value']]).reset_index(drop=True)
            else:
                resid = pd.DataFrame(sdt_emp.loc[sdt_emp['variable']=='c', ['subj_idx', 'bin', 'value']]).reset_index(drop=True)
            resid['value'] = (resid['value'] - np.array(sdt_sim.loc[sdt_emp['variable']=='c', 'value']))**2
            resid['model'] = version
            resids.append(resid)

            # bics:
            bic = pd.DataFrame(params.loc[params['variable']=='bic', ['subj_idx', 'value']]).reset_index(drop=True)
            bic['model'] = version
            bics.append(bic)

        if (version == 4) | (version == 9):

            # SDT analysis:
            colors = sns.color_palette(n_colors=3)
            colors = [colors[2], colors[0], colors[1]]
            if experiment_name == 'bias_manipulation':
                for (c), d in df_emp.groupby(['cons']):
                    sdt_sims_ = []
                    for sdt_sim in sdt_sims:
                        sdt_sims_.append(sdt_sim.loc[(sdt_sim['cons']==c),:])
                    fig = mixed_linear_modeling(sdt_emp.loc[(sdt_emp['cons']==c),:], x='bin', df_sims=sdt_sims_, colors=colors)
                    fig.savefig(os.path.join(project_dir, 'figs', 'ddm', '{}_{}_sdt_bars_{}.pdf'.format(experiment_name, version, c)))
            else:
                fig = analyses_tools.mixed_linear_modeling(sdt_emp, x='bin', df_sims=sdt_sims, colors=colors)
                fig.savefig(os.path.join(project_dir, 'figs', 'ddm', '{}_{}_sdt_bars.pdf'.format(experiment_name, version)))

            # BICs & residuals:
            bics = pd.concat(bics)
            resids = pd.concat(resids)
            # resids = resids.groupby(['subj_idx', 'model']).mean().reset_index()
            print()
            print(experiment_name)
            print(bics.groupby('model').mean())        
            print(resids.groupby('model').mean())

            # subtract bics:
            subtract = np.array(bics.loc[bics['model']==bics['model'].min(), 'value'])
            for m in bics['model'].unique():
                bics.loc[bics['model']==m, 'value'] = np.array(bics.loc[bics['model']==m, 'value']) - subtract
            
            # subtract resids:
            subtract = np.array(resids.loc[resids['model']==resids['model'].min(), 'value'])
            for m in resids['model'].unique():
                resids.loc[resids['model']==m, 'value'] = np.array(resids.loc[resids['model']==m, 'value']) - subtract
            
            for data, title in zip([bics, resids], ['bic', 'resid']):

                fig = plt.figure(figsize=(1.5,1.5))
                sns.barplot(x='model', y='value', units='subj_idx', palette=colors, ci=66, data=data)
                # sns.stripplot(x='model', y='value', color='grey', size=2, data=data)
                if not title == 'bic':
                    for i, m in enumerate(versions):
                        t,p = sp.stats.ttest_1samp(resids.loc[resids['model']==m,'value'],0)   
                        plt.text(i, plt.gca().get_ylim()[1]-((plt.gca().get_ylim()[1]-plt.gca().get_ylim()[0])/10), 'p={}'.format(round(p,3)), size=5, rotation=45)

                plt.xticks([0,1,2], ['z', 'dc', 'u'])
                plt.ylabel(title)
                sns.despine(offset=2, trim=True)
                plt.tight_layout()
                fig.savefig(os.path.join(project_dir, 'figs', 'ddm', '{}_{}_{}.pdf'.format(experiment_name, version, title)))

                fig = plt.figure(figsize=(1.5,1.5))
                if title == 'bic':
                    sns.barplot(x='model', y='value', units='subj_idx', ci=66, errwidth=1, palette=colors, data=data)
                    plt.xticks([0,1,2], ['z', 'dc', 'u'])
                if title == 'resid':
                    sns.barplot(x='bin', y='value', hue='model', units='subj_idx', ci=66, errwidth=1, palette=colors, data=data)
                    aovrm = AnovaRM(data, 'value', 'subj_idx', within=['model','bin'], aggregate_func='mean')
                    res = aovrm.fit()
                    print(res)
                    plt.title('p = {}'.format(round(res.anova_table.iloc[0]['Pr > F'],3)))
                    # plt.xticks([0,1,2], ['z', 'dc', 'u'])
                plt.ylabel(title)
                try:
                    plt.gca().get_legend().remove()
                except:
                    pass
                sns.despine(offset=2, trim=True)
                plt.tight_layout()
                fig.savefig(os.path.join(project_dir, 'figs', 'ddm', '{}_{}_{}2.pdf'.format(experiment_name, version, title)))