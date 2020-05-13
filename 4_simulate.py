#!/usr/bin/env python
# encoding: utf-8

import os
import numpy as np
import scipy as sp
import matplotlib as mpl
mpl.use("TkAgg")
mpl.rcParams['pdf.fonttype'] = 42

import matplotlib.pylab as plt
import seaborn as sns
import pandas as pd
from IPython import embed as shell

import accumodels
from accumodels import sim_tools
from accumodels.sim_tools import get_DDM_traces, apply_bounds_diff_trace, _bounds, _bounds_collapse_linear, _bounds_collapse_hyperbolic
from accumodels.plot_tools import summary_plot_group, conditional_response_plot
from tqdm import tqdm

sns.set(style='ticks', font='Arial', font_scale=1, rc={
    'axes.linewidth': 0.25, 
    'axes.labelsize': 7, 
    'axes.titlesize': 7, 
    'xtick.labelsize': 6, 
    'ytick.labelsize': 6, 
    'legend.fontsize': 6, 
    'xtick.major.width': 0.25, 
    'ytick.major.width': 0.25,
    'text.color': 'Black',
    'axes.labelcolor':'Black',
    'xtick.color':'Black',
    'ytick.color':'Black',} )
sns.plotting_context()

def do_simulations(params):
    rt = []
    response = []
    stimulus = []
    for stim in [1,0]:
        
        # get traces:
        x = get_DDM_traces(v=params['v'],
                            z=params['z'],
                            dc=params['dc'],
                            dc_slope=params['dc_slope'],
                            sv=params['sv'],
                            stim=stim,
                            nr_trials=params['nr_trials'],
                            tmax=tmax,
                            dt=dt,)
        
        # get bounds:
        if params['bound'] == 'default':
            b1, b0 = _bounds(a=params['a'], tmax=tmax, dt=dt)
        elif params['bound'] == 'collapse_linear':
            b1, b0 = _bounds_collapse_linear(a=params['a'], c1=params['c1'], c0=params['c0'], tmax=tmax, dt=dt)
        elif params['bound'] == 'collapse_hyperbolic':
            b1, b0 = _bounds_collapse_hyperbolic(a=params['a'], c=params['c'], tmax=tmax, dt=dt)
        
        # apply bounds:
        rt_dum, response_dum = apply_bounds_diff_trace(x=x, b1=b1, b0=b0)
        
        # store results:
        rt.append((rt_dum*dt)+ndt)
        response.append(response_dum)
        stimulus.append(np.ones(params['nr_trials']) * stim)

    df = pd.DataFrame()
    df.loc[:,'rt'] = np.concatenate(rt)
    df.loc[:,'response'] = np.concatenate(response)
    df.loc[:,'stimulus'] = np.concatenate(stimulus)
    df.loc[:,'correct'] = np.array(np.concatenate(stimulus) == np.concatenate(response), dtype=int)
    df.loc[:,'subj_idx'] = params['subj_idx']
    df.to_csv(os.path.join(data_folder, 'df_{}.csv'.format(params['subj_idx'])))


data_folder = os.path.expanduser('/home/jwdegee/repos/2020_eLife_pupil_bias/data/simulate')
fig_folder = os.path.expanduser('/home/jwdegee/repos/2020_eLife_pupil_bias/figs/simulate')

simulate = True
parallel = True
nr_trials = int(1e5) #100K
# nr_trials = int(1e4) #10.000
tmax = 5
dt = 0.01

v = 0.5
a = 2
dc = 0
dc_slope = 0
ndt = 0.3
sv = 0

sArray = [

    # 1 increasing starting point bias:
    {'subj_idx':0, 'v':v, 'dc':dc, 'z':0.50*a, 'a':a, 'dc_slope':dc_slope, 'sv':sv, 'bound':'default', 'nr_trials':nr_trials},
    {'subj_idx':1, 'v':v, 'dc':dc, 'z':0.55*a, 'a':a, 'dc_slope':dc_slope, 'sv':sv, 'bound':'default', 'nr_trials':nr_trials},
    {'subj_idx':2, 'v':v, 'dc':dc, 'z':0.60*a, 'a':a, 'dc_slope':dc_slope, 'sv':sv, 'bound':'default', 'nr_trials':nr_trials},

    # increasing drift bias:
    {'subj_idx':3, 'v':v, 'dc':dc+0.00, 'z':0.5*a, 'a':a, 'dc_slope':dc_slope, 'sv':sv, 'bound':'default', 'nr_trials':nr_trials},
    {'subj_idx':4, 'v':v, 'dc':dc+0.12, 'z':0.5*a, 'a':a, 'dc_slope':dc_slope, 'sv':sv, 'bound':'default', 'nr_trials':nr_trials},
    {'subj_idx':5, 'v':v, 'dc':dc+0.24, 'z':0.5*a, 'a':a, 'dc_slope':dc_slope, 'sv':sv, 'bound':'default', 'nr_trials':nr_trials},

    # fixed starting point bias, increasing collapsing bounds:
    {'subj_idx':6, 'v':v, 'dc':dc+0, 'z':0.55*a, 'a':a, 'dc_slope':dc_slope, 'sv':sv, 'bound':'collapse_hyperbolic', 'c':200.0, 'nr_trials':nr_trials},
    {'subj_idx':7, 'v':v, 'dc':dc+0, 'z':0.55*a, 'a':a, 'dc_slope':dc_slope, 'sv':sv, 'bound':'collapse_hyperbolic', 'c':2, 'nr_trials':nr_trials},
    {'subj_idx':8, 'v':v, 'dc':dc+0, 'z':0.55*a, 'a':a, 'dc_slope':dc_slope, 'sv':sv, 'bound':'collapse_hyperbolic', 'c':0.5, 'nr_trials':nr_trials},

    # fixed starting point bias, increasing collapsing bounds:
    {'subj_idx':9, 'v':v, 'dc':dc+0.12, 'z':0.5*a, 'a':a, 'dc_slope':dc_slope, 'sv':sv, 'bound':'collapse_hyperbolic', 'c':200.0, 'nr_trials':nr_trials},
    {'subj_idx':10, 'v':v, 'dc':dc+0.12, 'z':0.5*a, 'a':a, 'dc_slope':dc_slope, 'sv':sv, 'bound':'collapse_hyperbolic', 'c':2, 'nr_trials':nr_trials},
    {'subj_idx':11, 'v':v, 'dc':dc+0.12, 'z':0.5*a, 'a':a, 'dc_slope':dc_slope, 'sv':sv, 'bound':'collapse_hyperbolic', 'c':0.5, 'nr_trials':nr_trials},

    # # 1 increasing starting point bias, fixed collapsing bounds:
    # {'subj_idx':6, 'v':v, 'dc':dc+0, 'z':0.5*a, 'a':a, 'dc_slope':dc_slope, 'sv':sv, 'bound':'collapse_hyperbolic', 'c':0.8, 'nr_trials':nr_trials},
    # {'subj_idx':7, 'v':v, 'dc':dc+0, 'z':0.53*a, 'a':a, 'dc_slope':dc_slope, 'sv':sv, 'bound':'collapse_hyperbolic', 'c':0.8, 'nr_trials':nr_trials},
    # {'subj_idx':8, 'v':v, 'dc':dc+0, 'z':0.56*a, 'a':a, 'dc_slope':dc_slope, 'sv':sv, 'bound':'collapse_hyperbolic', 'c':0.8, 'nr_trials':nr_trials},

    ]

# up1, down1 = _bounds_collapse_hyperbolic(2, 1.6, lower_is_0=True, tmax=5, dt=0.01)
# up2, down2 = _bounds_collapse_hyperbolic(2, 0.8, lower_is_0=True, tmax=5, dt=0.01)
# up3, down3 = _bounds_collapse_hyperbolic(2, 0.4, lower_is_0=True, tmax=5, dt=0.01)
# plt.plot(up1, color='r')
# plt.plot(down1, color='r')
# plt.plot(up2, color='g')
# plt.plot(down2, color='g')
# plt.plot(up3, color='b')
# plt.plot(down3, color='b')

if simulate:
    if not parallel:
        for i, s in tqdm(enumerate(sArray)):
            do_simulations(s) 
    else:
        from joblib import Parallel, delayed
        n_jobs = 42
        res = Parallel(n_jobs=n_jobs)(delayed(do_simulations)(params) for params in sArray)
        # do_simulations(sArray[0])


groups = [list(np.arange(0,3)), list(np.arange(3,6)), list(np.arange(6,9)), list(np.arange(9,12)),]
quantiles = [0, 0.1, 0.3, 0.5, 0.7, 0.9, 1]

cmaps = ["Greens", 'Blues', 'Oranges', 'Purples', 'RdPu']

for i, group in enumerate(groups):
    
    # neutral:
    df = pd.read_csv(os.path.join(data_folder, 'df_{}.csv'.format(0)))
    mean_correct = df.correct.mean()
    mean_response = df.response.mean()
    
    # load group:
    df = pd.concat([pd.read_csv(os.path.join(data_folder, 'df_{}.csv'.format(g))) for g in group], axis=0)
    
    # plots:
    quantiles = [0, 0.1, 0.3, 0.5, 0.7, 0.9,]

    fig = conditional_response_plot(df, quantiles, mean_response, xlim=(0.2,2), cmap=cmaps[i])
    fig.savefig(os.path.join(fig_folder, 'crf_{}.pdf'.format(i)))

    fig = conditional_response_plot(df, quantiles, mean_response, xlim=(0.2,2), cmap=cmaps[i])
    fig.savefig(os.path.join(fig_folder, 'crf_{}.pdf'.format(i)))

    fig = conditional_response_plot(df, quantiles, mean_response, y='correct', xlim=(0.2,2), ylim=[0.5,0.8], cmap=cmaps[i])
    fig.savefig(os.path.join(fig_folder, 'caf_{}.pdf'.format(i)))

    for s, d in df.groupby(['subj_idx']):

        fig = summary_plot_group(d, df_sim_group=d, quantiles=quantiles, step_size=0.01, xlim=(0.2,2))
        fig.savefig(os.path.join(fig_folder, 'summary_{}_{}.pdf'.format(i,s)))

    # fig = summary_plot(df, quantiles, mean_correct, mean_response, xlim=(0.1,0.7))
    # fig.savefig(os.path.join(fig_folder, 'summary_{}.pdf'.format(i)))

    print(df.groupby('subj_idx').mean())