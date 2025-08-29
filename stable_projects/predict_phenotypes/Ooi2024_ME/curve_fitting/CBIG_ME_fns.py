#!/usr/bin/env python
# coding: utf-8
'''
This module contains functions relevant to curve fitting and figure
plotting in Ooi2024_ME.

There are 4 sections:
1) data loading functions
2) figure plotting functions
3) curve fitting functions
4) stats functions

Written by Leon Ooi and CBIG under MIT license:
https://github.com/ThomasYeoLab/CBIG/blob/master/LICENSE.md
'''

############################################
# IMPORT DEPENDENCIES
############################################
import numpy as np
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy
import scipy.io
import pickle
import sys
import os
from mpl_toolkits.axes_grid1 import make_axes_locatable


############################################
# DATA LOADING FUNCTIONS
############################################
def load_data(dataset,
              analysis,
              rep_dir,
              vers='full',
              reg='KRR',
              metric='corr'):
    """
    Load collated output files for specified analysis and generate
    related figure settings.

    Inputs:
        dataset       - Dataset to load.
        analysis      - Data to be loading. Can be 'predacc',
                        'tstats', 'pfrac' or 'Haufe'
        rep_dir       - Directory in which results are stored
        vers          - FC generation method. Can be 'full', 'random',
                        'uncensored_only', 'no_censoring'
        reg           - Regression method that was used. Either
                        'KRR' or 'LRR'.
        metric        - Accuracy metric used. Can be 'corr' or 'COD'

    Outputs:
        img_dir       - Output directory for images (creates a
                        folder in rep_dir)
        res           - Matrix of data that was loaded
        X             - Time values for the dataset
        Y             - Subject values for the dataset
        extent        - Limits of X and Y (used to plot contour plot)
        scan_duration - Matrix of total scan time that matches res
    """
    # Parameters for loading prediction accuracy results
    if analysis == 'predacc':
        output_vers = 'output'
        # read output files
        img_dir = os.path.join(rep_dir, dataset, output_vers, vers, 'images')
        mat = os.path.join(img_dir,
                           'acc_' + reg + '_' + metric + '_landscape.mat')
        res = scipy.io.loadmat(mat)
        # set plotting variables
        if dataset == 'HCP':
            Y = np.array([200, 300, 400, 500, 600, 700])
            X = np.linspace(2, 58, num=29, dtype=int)
            extent = [2, 58, 200, 700]
        elif dataset == 'ABCD':
            if ("MID" in vers) or ("NBACK" in vers) or ("SST" in vers):
                Y = np.array([200, 400, 600, 800, 1000, 1200, 1400, 1600])
                if ("MID" in vers):
                    max_X = 11
                if ("NBACK" in vers):
                    max_X = 10
                if ("SST" in vers):
                    max_X = 12
                X = np.linspace(2, max_X, num=(max_X - 1), dtype=int)
                extent = [2, max_X, 200, 1600]
            else:
                Y = np.array(
                    [200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800])
                X = np.linspace(2, 20, num=10, dtype=int)
                extent = [2, 20, 200, 1800]
        elif dataset == 'SINGER':
            Y = np.array([100, 200, 300, 400, 500, 580])
            X = np.linspace(2, 10, num=9, dtype=int)
            extent = [2, 10, 100, 580]
        elif dataset == 'TCP':
            Y = np.array([50, 75, 100, 125, 150, 175])
            X = np.linspace(2, 26, num=13, dtype=int)
            extent = [2, 26, 50, 175]
        elif dataset == 'ADNI':
            Y = np.array([100, 200, 300, 400, 500])
            X = np.linspace(2, 9, num=8, dtype=int)
            extent = [2, 9, 100, 500]
        elif dataset == 'MDD':
            Y = np.array([50, 75, 100, 125, 150, 175, 200, 225, 250, 260])
            X = np.linspace(3, 23, num=11, dtype=int)
            extent = [3, 23, 50, 260]
        else:
            sys.exit('dataset not recognised!')
        scan_duration = np.tile(Y[:, np.newaxis], (1, len(X))) * X

    # Parameters for loading reliability results
    elif (analysis == 'predacc_sh' or analysis == 'tstats'
          or analysis == 'Haufe' or analysis == 'pfrac'):
        output_vers = 'output_splithalf'
        # read output files
        img_dir = os.path.join(rep_dir, dataset, output_vers, vers, 'images')
        if analysis == 'tstats':
            mat = os.path.join(img_dir, 'tstats_icc_landscape.mat')
        elif analysis == 'pfrac':
            mat = os.path.join(img_dir, 'p_frac_landscape.mat')
        elif analysis == 'Haufe':
            mat = os.path.join(img_dir, 'fi_icc_KRR_landscape.mat')
        elif analysis == 'predacc_sh':
            mat = os.path.join(img_dir,
                               'acc_' + reg + '_' + metric + '_landscape.mat')
        res = scipy.io.loadmat(mat)
        # set plotting variables
        if dataset == 'HCP':
            Y = np.array([150, 200, 250, 300, 350, 400])
            X = np.linspace(2, 58, num=29, dtype=int)
            extent = [2, 58, 150, 400]
        elif dataset == 'ABCD':
            Y = np.array([200, 400, 600, 800, 1000, 1200])
            X = np.linspace(2, 20, num=10, dtype=int)
            extent = [2, 20, 200, 1200]
        else:
            sys.exit('Dataset not recognised!')
        scan_duration = np.tile(Y[:, np.newaxis], (1, len(X))) * X
    else:
        sys.exit('Analysis not recognised!')
    return img_dir, res, X, Y, extent, scan_duration


def load_fits(dataset, analysis, rep_dir, vers='full'):
    """
    Load collated output files for specified analysis and
    generate related figure settings.

    Inputs:
        dataset       - Dataset to load.
        analysis      - Data to be loading. Can be 'predacc', 'tstats',
                        'pfrac' or 'Haufe'
        rep_dir       - Directory in which results are stored
        vers          - FC generation method. Can be 'full', 'random',
                        'uncesored_only', 'no_censoring'

    Outputs:
        w_r_all       - All weights for reliability eqn
        w_pa_all      - All weights for prediction accuracy eqn
        zk_all        - All coefficients for log eqn
        loss_r_all    - COD loss for reliability eqn
        loss_pa_all   - COD loss for prediction accuracy eqn
        loss_log_all  - COD loss for log eqn

    """
    # set number of phenotypes based on dataset
    if dataset == 'HCP':
        num_behavs = 61
        test_length = 27
    elif dataset == 'ABCD':
        num_behavs = 39
        if ("MID" in vers):
            test_length = 8
        elif ("NBACK" in vers):
            test_length = 7
        elif ("SST" in vers):
            test_length = 9
        else:
            test_length = 8
    elif dataset == 'SINGER':
        num_behavs = 19
        test_length = 7
    elif dataset == 'HCP_7T':
        num_behavs = 60
        test_length = 28
    elif dataset == 'TCP':
        num_behavs = 19
        test_length = 11
    elif dataset == 'ADNI':
        num_behavs = 7
        test_length = 6
    elif dataset == 'MDD':
        num_behavs = 20
        test_length = 9
    else:
        sys.exit('dataset not recognised!')
    # Initialize variables to be saved
    w_r_all = np.zeros((num_behavs, test_length, 3))
    w_pa_all = np.zeros((num_behavs, test_length, 3))
    zk_all = np.zeros((num_behavs, test_length, 2))
    loss_r_all = np.zeros((num_behavs, test_length))
    loss_pa_all = np.zeros((num_behavs, test_length))
    loss_log_all = np.zeros((num_behavs, test_length))
    # Parameters for loading prediction accuracy fits
    if analysis == 'predacc':
        output_vers = 'output'
        # read output files
        curve_dir = os.path.join(rep_dir, dataset, output_vers, vers,
                                 'curve_fit')
        for b in range(num_behavs):
            pickle_path = os.path.join(
                curve_dir, analysis + '_behav' + str(b) + '_results.sav')
            pickle_f = open(pickle_path, 'rb')
            res = pickle.load(pickle_f)
            pickle_f.close()
            w_r_all[b, :, :] = res['w_r_sav']
            w_pa_all[b, :, :] = res['w_pa_sav']
            zk_all[b, :, :] = res['zk_sav']
            loss_r_all[b, :] = res['loss_n_r']
            loss_pa_all[b, :] = res['loss_n_pa']
            loss_log_all[b, :] = res['loss_log']

    # Parameters for loading reliability results
    elif analysis == 'tstats' or analysis == 'Haufe' or analysis == 'pfrac':
        output_vers = 'output_splithalf'
        # read output files
        curve_dir = os.path.join(rep_dir, dataset, output_vers, vers,
                                 'curve_fit')
        for b in range(num_behavs):
            pickle_path = os.path.join(
                curve_dir, analysis + '_behav' + str(b) + '_results.sav')
            pickle_f = open(pickle_path, 'rb')
            res = pickle.load(pickle_f)
            pickle_f.close()
            w_r_all[b, :, :] = res['w_r_sav']
            w_pa_all[b, :, :] = res['w_pa_sav']
            zk_all[b, :, :] = res['zk_sav']
            loss_r_all[b, :] = res['loss_n_r']
            loss_pa_all[b, :] = res['loss_n_pa']
            loss_log_all[b, :] = res['loss_log']
    return w_r_all, w_pa_all, zk_all, loss_r_all, loss_pa_all, loss_log_all


############################################
# PLOTTING FUNCTIONS
############################################
def plot_contour(score,
                 X,
                 Y,
                 con_lines,
                 manual_loc,
                 extent,
                 fig,
                 ax,
                 default_font_sz=12,
                 Yax_lbl='Correlation',
                 Ax_Ttl=None):
    """
    Plot contour plot of participants and scan time against accuracy in a
    specified figure and axis.

    Inputs:
        score           - Matrix of data that was loaded. Can be prediction
                          accuracies or reliability values.
        X               - Time (T) values for the dataset
        Y               - Subject (N) values for the dataset
        con_lines       - List of values to place contour lines at
        manual_loc      - Location to display the numerical value of the
                          contour
        extent          - A grid which corresponds to the limits of N and T
                          that are drawn
        fig             - Figure object in which contour plots will be drawn in
        ax              - Axis object to draw figure (pass in the subplot axis)
        default_font_sz - Font size for ticks and labels in the plot.
                          Set to 12 by default.
        Yax_lbl         - Colorbar label. Set to "Correlation" by default.
        Ax_Ttl          - A string containing the title of the plot.

    Outputs:
        Generate countour plot in the specified fig and ax.
    """

    # resample data so that contour plots appear larger
    def f(X_in, Y_in):
        return score[int(Y_in), int(X_in)]

    g = np.vectorize(f)
    # resample grid with factor of 100
    X_grid = np.linspace(0, score.shape[1], score.shape[1] * 100)
    Y_grid = np.linspace(0, score.shape[0], score.shape[0] * 100)
    X_upsample, Y_upsample = np.meshgrid(X_grid[:-1], Y_grid[:-1])
    score_resample = np.flip(np.flip(g(X_upsample[:-1], Y_upsample[:-1])), 1)

    # plot contour plot
    extent = [
        0 - 0.5, X_upsample[:-1].max() - 0.5, 0 - 0.5,
        Y_upsample[:-1].max() - 0.5
    ]
    # rounding the values and adding offset makes the contour look nicer
    contours = ax.contour(
        np.round(score_resample[::-1] + 0.006, 3),
        con_lines,
        extent=extent,
        colors='black')
    ax.clabel(
        contours, inline=True, fontsize=default_font_sz, manual=manual_loc)
    # overlay colours
    c_c = ax.imshow(
        score_resample[::-1],
        extent=extent,
        origin='lower',
        aspect='auto',
        cmap='plasma',
        alpha=0.5)

    # update plot settings
    # update color bar
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    cbar = fig.colorbar(c_c, cax=cax, orientation='vertical')
    tick_locator = mpl.ticker.MaxNLocator(nbins=4)
    cbar.locator = tick_locator
    cbar.update_ticks()
    cbar.ax.tick_params(labelsize=default_font_sz)
    # update labels
    if Ax_Ttl:
        ax.set_title(Ax_Ttl, fontsize=default_font_sz)
    ax.set_xlabel(
        'Scan Time Per Participant',
        fontsize=default_font_sz,
        fontname='Arial')
    ax.locator_params(axis='x', nbins=10)
    if Yax_lbl == 'Correlation':
        ax.set_ylabel(
            '# Training Participants',
            fontsize=default_font_sz,
            fontname='Arial')
    else:
        ax.set_ylabel(
            '# Participants', fontsize=default_font_sz, fontname='Arial')


def plot_scatter(num_lvls,
                 beh_all,
                 scan_duration,
                 color_scheme,
                 limit,
                 ax,
                 outline='N'):
    """
    Plot scatter plot of results against total scan time. Use different
    color for each number of participants.

    Inputs:
        num_lvls      - How many levels of participants to plot
        beh_all       - Matrix of data that was loaded. Can be
                        prediction accuracies or reliability values.
        scan_duration - Values to use on Y axis (Total scan time or
                        scan time per participant)
        color_scheme  - List of colors to use when plotting different
                        levels of participants
        limit         - Plot scatter plot up to this value.
        ax            - Axis object to draw figure in (pass in the
                        subplot axis)
        outline       - Highlight the dots with a black outline.
                        Either 'Y' or 'N'.

    Outputs:
        Figure of scatter plot is produced. It can be specified whether
        to draw dots after limit with black outlines.
    """
    # plot from smallest subject cohort to biggest
    for n_subs in range(0, num_lvls):
        if outline == 'Y':
            # plot dots after limit with black outlines
            beh = beh_all[n_subs, limit:]
            curr_scan = scan_duration[n_subs, limit:]
            sns.scatterplot(
                x=curr_scan.flatten(),
                y=beh.flatten(),
                ax=ax,
                color=color_scheme[n_subs],
                edgecolor="k",
                linewidth=0.75,
                s=30)
        else:
            # plot dots before limit without black outlines
            beh = beh_all[n_subs, :limit]
            curr_scan = scan_duration[n_subs, :limit]
            sns.scatterplot(
                x=curr_scan.flatten(),
                y=beh.flatten(),
                ax=ax,
                color=color_scheme[n_subs])


def plot_norm_scatter(dataset,
                      analysis,
                      rep_dir,
                      fc_vers,
                      limit,
                      inds,
                      color_scheme,
                      log_scale=False,
                      zorder=None):
    """
    Load saved results and models fits to plot normalized scatter plot
    of results against total scan time. Use different shade for each
    phenotype.

    Inputs:
        dataset       - Dataset to load.
        analysis      - Data to be loading. Can be 'predacc',
                        'tstats', 'pfrac' or 'Haufe'
        rep_dir       - Directory in which results are stored
        fc_vers       - FC generation method. Can be 'full', 'random',
                        'uncensored_only', 'no_censoring'
        limit         - A scalar containing an index of the point after which
                        scan duration will be ignored
        inds          - array containing phenotype indices to plot
        color_scheme  - List of colors to use when plotting phenotypes
        log_scale     - True or False. Whether to plot time on a log scale.
        zorder        - Customize order of scatter plots (if necessary)

    Outputs:
        Figure of scatter plot on a normalized scale is produced.
    """
    # load data from dataset
    img_dir, res, X, Y, extent, scan_duration = load_data(
        dataset, analysis, rep_dir, vers=fc_vers)
    w_r, w_pa, zk, loss_r, loss_pa, loss_log = load_fits(
        dataset, analysis, rep_dir, vers=fc_vers)
    # iterate over each index passed in
    n_c = 0
    for n in inds:
        # extract saved parameters for specific phenotype
        behav = np.flip(np.flip(res['acc_landscape'][:, :, n].T), 1)
        if log_scale:
            curr_scan = np.log(scan_duration[:, :limit]) / np.log(2)
        else:
            curr_scan = scan_duration[:, :limit]
        behav = behav[:, :limit]
        z = zk[n, limit - 3, 0]
        k = zk[n, limit - 3, 1]
        # normalize accuracy
        norm_acc = (behav - k) / z
        # produce scatter plot
        if zorder:
            sns.scatterplot(
                x=curr_scan.flatten(),
                y=norm_acc.flatten(),
                color=color_scheme[n_c],
                zorder=zorder)
        else:
            sns.scatterplot(
                x=curr_scan.flatten(),
                y=norm_acc.flatten(),
                color=color_scheme[n_c])
        n_c += 1


def plot_curve(min_t, max_t):
    """
    Plot log curve with black outline.

    Inputs:
        min_t         - Value to start curve from
        max_t         - Value to stop curve at

    Outputs:
        Figure of log plot is produced. Remember to save the
        figure after adding additional plots (e.g. curve fits).
    """
    # set curve for specified limits
    X_fit = np.linspace(min_t, max_t, num=100, dtype=int)
    curve_val = np.log(X_fit) / np.log(2)
    # plot
    plt.plot(X_fit, curve_val, color='k')


def format_scatter_plot(x_lbl, y_lbl, ax, fontsz=12):
    """
    Format figures by changing fontsizes and removing spines

    Inputs:
        x_lbl         - x labels to show in the figure
        y_lbl         - y labels to show in the figure
        ax            - axes of figure to be modified
        fontsz        - font size of axes

    Outputs:
        Figure will be modified accordingly.
    """
    # format labels
    ax.set_xlabel(x_lbl, fontsize=fontsz, fontname='Arial')
    ax.set_ylabel(y_lbl, fontsize=fontsz, fontname='Arial')
    ax.xaxis.set_tick_params(labelsize=fontsz)
    ax.yaxis.set_tick_params(labelsize=fontsz)
    # remove spines
    ax.spines[['right', 'top']].set_visible(False)


def calc_datasetAcc(budget,
                    scanner_cost,
                    recruitment_cost,
                    T,
                    ind,
                    dataset,
                    rep_dir,
                    c_vers,
                    trainingsize=0.9,
                    rd=None):
    """
    Calculate the average fraction of maximum accuracy for a
    dataset given a specified set of phenotypes.

    Inputs:
        budget            - an integer representing total
                            fMRI budget
        scanner_cost      - an integer representing cost
                            per hour of MRI scan
        recruitment_cost  - an integer representing cost
                            of recruiting each participant
        T                 - vector representing the range of
                            scan times to calculate accuracy for
        ind               - indices of phenotypes to use
        dataset           - dataset to calculate for
        rep_dir           - location where data is saved
        c_vers            - version of analysis
        trainingsize      - fraction of data used for training
                            set. 0.9 by default.
        rd                - whether to round down participants to
                            get whole numbers

    Outputs:
        mean_acc          - mean accuracy over all specified
                            phenotypes for T time points
        c_int             - the confidence interval over phenotypes
    """
    # calculate number of subjects that can be afforded
    if rd:
        N = np.floor(budget / (T * scanner_cost + recruitment_cost))
    else:
        N = budget / (T * scanner_cost + recruitment_cost)
    training_N = trainingsize * N
    if rd:
        training_N = np.floor(training_N)

    # load data
    w_r_all, w_pa_all, zk_all, loss_r_all, loss_pa_all, loss_log_all = \
        load_fits(dataset, 'predacc', rep_dir, vers=c_vers)
    # save prediction values for each phenotype
    theor_vals = []
    for b in ind:
        # Theoretical equation fit to full duration
        w = w_pa_all[b, -1, :]
        b_acc = np.sqrt(
            1 / (1 + (w[1] / training_N) + (w[2] / (training_N * T))))
        theor_vals.append(b_acc)

    # get confidence interval of prediction accuracy
    c_int = 1.96 * np.std(theor_vals) / np.sqrt((len(theor_vals)))

    return np.mean(theor_vals, 0), c_int


def calc_avgHCPABCDAcc(N,
                       T,
                       vers,
                       rep_dir,
                       HCP_behav_ind,
                       ABCD_behav_ind,
                       trainingsize=0.9,
                       rd=None):
    """
    Calculate the average fraction of maximum accuracy for a
    specified set of phenotypes within the ABCD and HCP only.

    Inputs:
        N                 - an integer representing total sample size
        T                 - vector representing the range of
        vers              - version of analysis
                            scan times to calculate accuracy for
        rep_dir           - location where data is saved
        HCP_behav_ind     - phenotype indices for HCP dataset
        ABCD_behav_ind    - phenotype indices for ABCD dataset
        trainingsize      - fraction of data used for training
                            set. 0.9 by default.
        rd                - whether to round down participants to
                            get whole numbers

    Outputs:
        mean_acc          - mean accuracy over all specified
                            phenotypes for T time points
        c_int             - the confidence interval over phenotypes
    """
    theor_vals = []
    # calculate number of subjects that can be afforded
    training_N = trainingsize * N
    if rd:
        training_N = np.floor(training_N)
    # load HCP results
    w_r_all, w_pa_all, zk_all, loss_r_all, loss_pa_all, loss_log_all = \
        load_fits('HCP', 'predacc', rep_dir, vers=vers)
    for b in HCP_behav_ind:
        # Tom's equation fit to full duration
        w = w_pa_all[b, -1, :]
        b_acc = np.sqrt(1 / (1 + (w[1] / N) + (w[2] / (N * T))))
        theor_vals.append(b_acc)

    # load ABCD results
    w_r_all, w_pa_all, zk_all, loss_r_all, loss_pa_all, loss_log_all = \
        load_fits('ABCD', 'predacc', rep_dir, vers=vers)
    for b in ABCD_behav_ind:
        # Tom's equation fit to full duration
        w = w_pa_all[b, -1, :]
        b_acc = np.sqrt(1 / (1 + (w[1] / N) + (w[2] / (N * T))))
        theor_vals.append(b_acc)

    # get confidence interval
    c_int = 1.96 * np.std(theor_vals) / np.sqrt((len(theor_vals)))

    return np.mean(theor_vals, 0), c_int


def calc_percOfmax(final_predacc, perc):
    """
    Find left and right limits that are within a specified percentage
    of the maximum accuracy.

    Inputs:
        final_predacc     - #T length vector of accuracy values
        perc              - percentage of maximum accuracy

    Outputs:
        curr_l            - index of left limit
        curr_r            - index of right limit
    """
    minacc = np.max(final_predacc) - perc / 100
    loc = 0
    curr_l = -1
    curr_r = -1
    for curr_predacc in final_predacc:
        if curr_predacc > minacc:
            if curr_l < 0:
                curr_l = loc
            else:
                curr_r = loc
        loc += 1
    return curr_l, curr_r


def plot_max_range(acc_vec, perc, T, c, ax):
    """
    Plot error bars in which the fraction of maximum accuracy is
    within specified percentage of the maximum accuracy. Error
    bar is set to be out of the window by 100 units if the right
    limit is at the maximum.

    Inputs:
        acc_vec     - #T length vector of accuracy values
        perc        - percentage of maximum accuracy
        T           - T-values corresponding to acc_vec
        c           - color of error bar
        ax          - axis on which to plot error bars on

    Outputs:
        Error bars plotted on specified axis.
    """
    # find interval
    lb, rb = calc_percOfmax(acc_vec, perc)
    # plot error bar and scatter plot
    x_val = T[np.argmax(acc_vec)]
    y_val = np.max(acc_vec)
    # Plotting error bars
    if rb < (len(acc_vec) - 1):
        plt.errorbar(
            x_val,
            y_val,
            xerr=[[x_val - T[lb]], [T[rb] - x_val]],
            capsize=4,
            ecolor=c,
            zorder=1)
    else:
        plt.errorbar(
            x_val,
            y_val,
            xerr=[[x_val - T[lb]], [T[rb] - x_val + 100]],
            capsize=4,
            ecolor=c,
            zorder=1)
    # plot optima (black outline if > maximum)
    if np.argmax(acc_vec) < (len(acc_vec) - 1):
        ax.scatter(
            T[np.argmax(acc_vec)],
            np.max(acc_vec),
            color=c,
            zorder=2,
            clip_on=True)


############################################
# CURVE FITTING FUNCTIONS
############################################
def calc_loss(y_true, y_pred):
    """
    Calculate the loss function for MSE and COD.

    Inputs:
        y_true        - vector of true values
        y_pred        - vector of predicted values

    Outputs:
        mse           - mean squared error loss
        COD           - coefficient of determination loss
    """
    mse = np.mean((y_true - y_pred)**2)
    RSS = np.sum((y_true - y_pred)**2)
    SST = np.sum((y_true - np.mean(y_true))**2)
    COD = 1 - RSS / SST
    return mse, COD


def lst_sq_log(total_time, beh):
    """
    Fit the following equation using least squares:
    beh = z (ln(total_time) / ln(2)) + k

    Inputs:
        total_time    - vector of total scan time
        beh           - vector of either accuracy or reliability values
                        for given behavior measure

    Outputs:
        z             - scaling factor for equation
        k             - offset for equation
    """
    # convert to log 2
    total_time = np.log(total_time) / np.log(2)
    # solve using least squares
    t_mat = np.vstack([total_time, np.ones(len(total_time))]).T
    z, k = np.linalg.lstsq(t_mat, beh, rcond=None)[0]
    return z, k


def gd_pred_acc(time,
                beh,
                ppts,
                learn_rate=1,
                n_iter=1000,
                tolerance=1e-06,
                verbose=0):
    """
    Fit the following equation using gradient descent:
    beh =  w[0] * np.sqrt(1/(1 + (w[1]/ppts) + (w[2]/(ppts*time))))

    Inputs:
        time          - vector of scan time (this is same as X from data
                        loading function)
        ppts          - vector of participants used (this is same as Y from
                        data loading function)
        beh           - vector of either accuracy or reliability values
                        for given behavior measure
        learn_rate    - learning rate for gradient descent
        n_iter        - maximum number of iterations for gradient descent
        tolerance     - exit gradient descent if change in weights are below
                        this number
        verbose       - print intermediate values during gradient descent

    Outputs:
        best_w        - best weights from gradient descent
    """
    # reshape inputs
    T = np.squeeze(np.tile(time, (1, len(ppts))))
    N = np.repeat(ppts, len(time))
    # initialize starting point
    loss = -1000
    best_loss = -1000
    init_w = [0.7, 0, 0]
    best_w = init_w
    # use grid search to find best starting point
    for w1 in range(0, 1500, 50):
        for w2 in range(0, 15000, 500):
            init_w[1] = w1
            init_w[2] = w2
            if verbose:
                print(init_w)
            w = np.array(init_w)

            # begin gradient descent
            for i in range(n_iter):
                # find change in weights - use mse loss as cost function
                denom = 1 / (1 + (w[1] / N) + (w[2] / (N * T)))
                dw0 = np.mean(
                    -2 * np.sqrt(denom) * (beh - (w[0] * np.sqrt(denom))))
                dw1 = np.mean(
                    w[0] * (denom**1.5) * (beh - (w[0] * np.sqrt(denom))) / N)
                dw2 = np.mean(
                    w[0] * (denom**1.5) * (beh -
                                           (w[0] * np.sqrt(denom))) / (N * T))
                # record final loss
                curve_val = w[0] * np.sqrt(1 / (1 + (w[1] / N) +
                                                (w[2] / (N * T))))
                mse, COD = calc_loss(beh.flatten(), curve_val)
                loss = -mse
                if verbose:
                    print("loss:", loss)
                # save best loss
                if loss > best_loss:
                    best_loss = loss
                    best_w = w
                # break if tolerance is reached
                diff = -learn_rate * np.array([dw0, dw1, dw2])
                if np.all(np.abs(diff) <= tolerance):
                    break
                # update weights
                w += diff
                if verbose and i == (n_iter - 1):
                    print('Max iterations reached!')
    return best_w


def gd_rel(time,
           beh,
           ppts,
           learn_rate=0.0001,
           n_iter=100,
           tolerance=1e-06,
           verbose=0):
    """
    Fit the following equation using gradient descent:
    beh =  w[0] / (w[0] + (1/(ppts/2)) * (1 - 2*w[1]/(1+(w[2]/time))))

    Inputs:
        time          - vector of scan time (this is same as X from data
                        loading function)
        ppts          - vector of participants used (this is same as Y from
                        data loading function)
        beh           - vector of either accuracy or reliability values
                        for given behavior measure
        learn_rate    - learning rate for gradient descent
        n_iter        - maximum number of iterations for gradient descent
        tolerance     - exit gradient descent if change in weights are below
                        this number
        verbose       - print intermediate values during gradient descent

    Outputs:
        best_w        - best weights from gradient descent
    """
    # reshape inputs
    T = np.squeeze(np.tile(time, (1, len(ppts))))
    N = np.repeat(ppts, len(time))
    # initialize starting point
    loss = -1000
    best_loss = -1000
    init_w = [0.001, 0.4, 1]
    best_w = init_w
    # use grid search to find best starting point
    for w0 in range(0, 30):
        for w1 in range(0, 10):
            for w2 in range(0, 10):
                init_w[0] = w0 / 20000
                init_w[1] = w1 / 10
                init_w[2] = w2
                if verbose:
                    print(init_w)
                w = np.array(init_w)

                # begin gradient descent
                for i in range(n_iter):
                    # find change in weights - use mse loss as cost function
                    common = w[2] + T - (2 * w[1] * T)
                    denom = (T * (w[0] * N - 4 * w[1] + 2) +
                             w[2] * (w[0] * N + 2))**3
                    dw0 = np.mean(
                        -4 * N * (w[2] + T) * common *
                        (w[0] * N * (beh - 1) *
                         (w[2] + T) + 2 * beh * common) / denom) * 0.001
                    dw1 = np.mean(-8 * w[0] * N * T * (w[2] + T) *
                                  (w[0] * N * (beh - 1) *
                                   (w[2] + T) + 2 * beh * common) / denom)
                    dw2 = np.mean(8 * w[0] * w[1] * N * T *
                                  (w[0] * N * (beh - 1) *
                                   (w[2] + T) + 2 * beh * common) / denom)
                    # record final loss
                    curve_val = w[0] / (w[0] + (1 / (N / 2)) *
                                        (1 - 2 * w[1] / (1 + (w[2] / T))))
                    mse, COD = calc_loss(beh.flatten(), curve_val)
                    loss = -mse
                    if verbose:
                        print("loss:", loss)
                    # save best loss
                    if loss > best_loss:
                        best_loss = loss
                        best_w = w
                    # break if tolerance is reached
                    diff = -learn_rate * np.array([dw0, dw1, dw2])
                    if np.all(np.abs(diff) <= tolerance):
                        break
                    # update weights
                    w += diff
                    if verbose and i == (n_iter - 1):
                        print('Max iterations reached!')
    return best_w


############################################
# STATS FUNCTIONS
############################################
def corrected_resample_ttest(accuracy_vec, portion, threshold):
    """
    Implementation of the corrected resampled t-test

    Inputs:
        accuracy_vec - vector of accuracy values
        portion      - ratio of training set to test set
        threshold    - mean to compare accuracy vector against

    Outputs:
        p            - p-value
    """
    K = len(accuracy_vec)

    # corrected variance
    corrected_variance = (1 / K + portion) * np.var(accuracy_vec)

    # tstat
    mu = np.mean(accuracy_vec)
    tval = np.divide((mu - threshold), np.sqrt(corrected_variance))

    # 2-tail p value (degree of freedom is K - 1)
    p = 2 * scipy.stats.t.sf(np.absolute(tval), df=K - 1)
    return p
