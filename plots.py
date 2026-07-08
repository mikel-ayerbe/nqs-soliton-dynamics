import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.transforms as mtransforms
from matplotlib.colors import TwoSlopeNorm
from matplotlib.ticker import ScalarFormatter

import numpy as np
import os
import string

import parameters as pm

# Enable LaTeX labels in Matplotlib
plt.rcParams['text.usetex'] = True
plt.rcParams['font.size'] = 12  # Optional: set the font size for clarity

def set_thesis_plot_style():
    """Matplotlib style matched to the 11 pt thesis text."""
    plt.rcParams.update({
        "text.usetex": True,
        "font.size": 11,
        "axes.labelsize": 11,
        "axes.titlesize": 11,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 11,
    })


def _to_numpy(a):
    """Convert torch tensors or array-like objects to NumPy arrays."""
    if hasattr(a, "detach"):
        return a.detach().cpu().numpy()
    return np.asarray(a)
def evo_fig_params_poster(time, mesh, den, params, fig_path='output.pdf'):
    
    # define variables
    t_max = max(time)
    real_params = np.real(params)
    imag_params = np.imag(params)
    num_params = params.shape[1]

    x_min = mesh[0] * 0.75  
    x_max = mesh[-1] * 0.75

    # Panel label
    ha = 'left'
    va = 'bottom'
    tx = 0.01
    ty = 0.05

    # Create the figure and GridSpec layout
    fig, axs = plt.subplots(3,1, 
                            figsize=(4, 2.5), 
                            height_ratios=[1,1,1],
                            sharex=True, 
                            layout='constrained')

    # Top Panel
    pcm_0 = axs[0].pcolor(time, mesh, den, 
                          cmap='viridis', 
                          shading='auto')
    axs[0].set_ylabel(r'$x/a_0$')
    axs[0].tick_params(labelbottom=False, length=5)
    axs[0].set_xticks(np.linspace(0, t_max, 6))
    axs[0].set_ylim([lim for lim in [x_min, x_max]])
    axs[0].set_yticks(np.linspace(x_min, x_max, 3))
    axs[0].text(tx, ty, r"${\rm (a)}$",
                horizontalalignment=ha,
                verticalalignment=va,
                transform=axs[0].transAxes,
                fontsize=12,
                color='white')

    # Add colorbar to the side without affecting panel width
    cbar_0 = fig.colorbar(pcm_0, ax=axs[0], pad=0, aspect=10, orientation='vertical')
    cbar_0.set_label(r'$|\psi(x,t)|^2$')
    cbar_0.set_ticks(np.round(np.linspace(0,max(map(max, den)),4),2))
    cbar_0.ax.yaxis.set_label_position('left')

    # Create a diverging colormap centered at zero
    cmap = 'RdBu' #'coolwarm'
    vabs = np.max([np.max(np.abs(real_params)), np.max(np.abs(imag_params))])
    norm = TwoSlopeNorm(vmin=-vabs, vcenter=0, vmax=vabs)

    # Middle Panel
    pcm_1 = axs[1].pcolor(time, 
                          np.arange(num_params) if num_params > 1 else np.arange(num_params + 1), 
                          real_params.T if num_params > 1 else np.hstack([real_params, real_params]).T,
                          cmap=cmap, 
                          norm=norm, 
                          shading='auto')
    axs[1].set_ylabel(r'$j$')
    axs[1].set_yticks(np.round(np.linspace(0, num_params-1, 3)))
    axs[1].tick_params(labelbottom=False, length=5)
    axs[1].set_xticks(np.linspace(0, t_max, 6))
    axs[1].text(tx,ty, r"${\rm (b)}$",
                horizontalalignment=ha,
                verticalalignment=va,
                transform=axs[1].transAxes,
                fontsize=12)
    
    # Bottom Panel
    axs[2].pcolor(time, 
                  np.arange(num_params) if num_params > 1 else np.arange(num_params + 1), 
                  imag_params.T if num_params > 1 else np.hstack([imag_params, imag_params]).T,
                  cmap=cmap, 
                  norm=norm, 
                  shading='auto')
    axs[2].set_xlabel(r'$t/\tau$')
    axs[2].set_ylabel(r'$j$')
    axs[2].tick_params(labelbottom=True, length=5)
    axs[2].set_xticks(np.round(np.linspace(0,max(time),6),1))
    axs[2].set_yticks(np.round(np.linspace(0, num_params-1, 3)))
    axs[2].text(tx, ty, r"${\rm (c)}$",
                horizontalalignment=ha,
                verticalalignment=va,
                transform=axs[2].transAxes,
                fontsize=12)
    
    # Add colorbar to the side without affecting panel width
    cbar_1 = fig.colorbar(pcm_1, ax=axs.ravel()[-2:].tolist(),
                          pad=0,
                          aspect=20,
                          orientation='vertical')
    cbar_1.ax.yaxis.set_label_position('left')
    cbar_1.set_ticks(np.round(np.linspace(-vabs*0.85, vabs*0.85, 5), 2))
    cbar_1.set_label(r'${\rm Im}[\theta_j] \;\quad\; {\rm Re}[\theta_j]$')

    

    # Adjust vertical spacing
    fig.get_layout_engine().set(w_pad=0, h_pad=0, hspace=0, wspace=0)

    # plt.show()

    # Ensure figs directory exists
    if not os.path.exists(pm.figs_dir):
        os.makedirs(pm.figs_dir)
    # Save the figure
    fig.savefig(fig_path, format=pm.fig_format, bbox_inches='tight', transparent=False)


def evo_fig_compare(time, mesh, den, energy, fig_path='compare.png'):
    
    # define variables
    t_max = max(time)

    ha = 'left'
    va = 'bottom'
    tx = 0.01
    ty = 0.05

    # Create the figure and GridSpec layout
    fig, axs = plt.subplots(2,1, figsize=(4, 3), sharex=True, layout='constrained')

    # Create a diverging colormap centered at zero
    cmap = 'RdBu' #'coolwarm'
    vabs = np.max(np.abs(den))
    norm = TwoSlopeNorm(vmin=-vabs, vcenter=0, vmax=vabs)

    # Top Panel
    pcm = axs[0].pcolor(time, mesh, den, cmap=cmap, norm=norm, shading='auto')
    axs[0].set_ylabel(r'$x/a_0$')
    axs[0].tick_params(labelbottom=False, length=5)
    axs[0].set_xticks(np.linspace(0, t_max, 6))
    axs[0].set_ylim([lim * 3 for lim in [-1.25, 1.25]])
    axs[0].set_yticks(np.linspace(-3, 3, 3))
    axs[0].text(tx, ty, r"${\rm (a)}$",
                horizontalalignment=ha,
                verticalalignment=va,
                transform=axs[0].transAxes,
                fontsize=12)

    # Add colorbar to the side without affecting panel width
    cbar = fig.colorbar(pcm, ax=axs[0], pad=0.01, aspect=10, orientation='vertical')
    cbar.set_label(r'$|\psi|^2-|\psi_0|^2$')
    # cbar.set_ticks(np.round(np.linspace(vmin, vmax, 5), 2))
    cbar.set_ticks(np.linspace(-vabs, vabs, 4))
    # Set colorbar ticks to scientific notation
    # formatter = ScalarFormatter(useMathText=True)
    # formatter.set_scientific(True)
    # formatter.set_powerlimits((-1, 1))
    formatter = ScalarFormatter(useMathText=True)
    formatter.set_powerlimits((0, 0))
    # formatter = FormatStrFormatter(r'$%.0e$')
    cbar.ax.yaxis.set_major_formatter(formatter)
    cbar.ax.yaxis.set_label_position('left')


    # Bottom Panel
    axs[1].plot(time, energy)
    axs[1].set_xlabel(r'$t/\tau$')
    axs[1].set_ylabel(r'$\delta E (\times 10^{-6})$')
    axs[1].tick_params(labelbottom=True, length=5)
    axs[1].set_xticks(np.round(np.linspace(0, t_max, 6), 1))
    axs[1].set_yticks(np.round(np.linspace(min(energy), max(energy), 3)))

    axs[1].text(tx,1-ty, r"${\rm (b)}$",
                horizontalalignment=ha,
                verticalalignment='top',
                transform=axs[1].transAxes,
                fontsize=12,
                )
        
    # Adjust vertical spacing
    fig.get_layout_engine().set(w_pad=0, h_pad=0, hspace=0, wspace=0)

    # plt.show()

    # Ensure figs directory exists
    if not os.path.exists(pm.figs_dir):
        os.makedirs(pm.figs_dir)
    # Save the figure
    fig.savefig(fig_path, format=pm.fig_format, bbox_inches='tight', transparent=False)

#-----------
def evo_fig_complete(mesh, t_imag, t_real, \
                    den_imag, den_real, \
                    params_imag, params_real, \
                    den_diff, energy, fig_path='fig_all.png'):
    
    
    # Enable LaTeX labels in Matplotlib
    plt.rcParams['text.usetex'] = True
    plt.rcParams['font.size'] = 17.5  # Optional: set the font size for clarity

    # Build the layout
    fig = plt.figure(constrained_layout=True, figsize=(12, 4))
    gs = gridspec.GridSpec(nrows=3, ncols=3, figure=fig)

    # First and second columns
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[2, 0])
    ax4 = fig.add_subplot(gs[0, 1])
    ax5 = fig.add_subplot(gs[1, 1])
    ax6 = fig.add_subplot(gs[2, 1])
    ax7 = fig.add_subplot(gs[0, 2])
    ax8 = fig.add_subplot(gs[1:3, 2])

    
    # Create a diverging colormap centered at zero
    cmap = 'RdBu' #'coolwarm'
    vabs = np.max([np.max(np.abs(params_real)), np.max(np.abs(params_imag))])
    norm = TwoSlopeNorm(vmin=-vabs, vcenter=0, vmax=vabs)

    # Imaginary evolution
    # define variables imag
    real_params = np.real(params_imag)
    imag_params = np.imag(params_imag)
    num_params = params_imag.shape[1]

    # Top Panel
    pcm_0 = ax1.pcolor(t_imag, mesh, den_imag, 
                            cmap='viridis', 
                            shading='auto')
    ax1.set_ylabel(r'$x$')

    # Middle Panel
    pcm_1 = ax2.pcolor(t_imag, 
                            np.arange(num_params) if num_params > 1 else np.arange(num_params + 1), 
                            real_params.T if num_params > 1 else np.hstack([real_params, real_params]).T,
                            cmap=cmap, 
                            norm=norm, 
                            shading='auto')

    # Bottom Panel
    ax3.pcolor(t_imag, 
                    np.arange(num_params) if num_params > 1 else np.arange(num_params + 1), 
                    imag_params.T if num_params > 1 else np.hstack([imag_params, imag_params]).T,
                    cmap=cmap, 
                    norm=norm, 
                    shading='auto')
    
    # Real evolution
    # define variables real
    real_params = np.real(params_real)
    imag_params = np.imag(params_real)
    num_params = params_imag.shape[1]

    # Top Panel
    pcm_0 = ax4.pcolor(t_real, mesh, den_real, 
                            cmap='viridis', 
                            shading='auto')
    ax1.set_ylabel(r'$x$')

    # Middle Panel
    pcm_1 = ax5.pcolor(t_real, 
                            np.arange(num_params) if num_params > 1 else np.arange(num_params + 1), 
                            real_params.T if num_params > 1 else np.hstack([real_params, real_params]).T,
                            cmap=cmap, 
                            norm=norm, 
                            shading='auto')

    # Bottom Panel
    ax6.pcolor(t_real, 
                    np.arange(num_params) if num_params > 1 else np.arange(num_params + 1), 
                    imag_params.T if num_params > 1 else np.hstack([imag_params, imag_params]).T,
                    cmap=cmap, 
                    norm=norm, 
                    shading='auto')

    # Add colorbar to the side without affecting panel width
    cbar_0 = fig.colorbar(pcm_0, ax=ax4, pad=0, aspect=10, orientation='vertical')
    cbar_0.set_label(r'$|\psi(x,t)|^2$')
    cbar_0.set_ticks(np.round(np.linspace(0,max(map(max, den_imag)),3),2))
    cbar_0.ax.yaxis.set_label_position('left')

    # Add colorbar to the side without affecting panel width
    cbar_1 = fig.colorbar(pcm_1, ax=[ax5, ax6],
                            pad=0,
                            aspect=20,
                            orientation='vertical')
    cbar_1.ax.yaxis.set_label_position('left')
    cbar_1.set_ticks(np.round(np.linspace(-vabs*0.85, vabs*0.85, 5), 2))
    cbar_1.set_label(r'${\rm Im}[\theta_k] \;\;\quad\;\; {\rm Re}[\theta_k]$')

    
    # Comparison    
    cmap = 'RdBu' #'coolwarm'
    vabs = np.max(np.abs(den_diff))
    norm = TwoSlopeNorm(vmin=-vabs, vcenter=0, vmax=vabs)
    # Top panel
    pcm_2 = ax7.pcolor(t_real, mesh, den_diff, 
                      cmap=cmap, norm=norm, shading='auto')
    # Bottom panel
    ax8.plot(t_real, energy)

    # Add colorbar to the side without affecting panel width    
    cbar = fig.colorbar(pcm_2, ax=ax7, pad=0.01, aspect=10, orientation='vertical')
    cbar.set_label(r'$\delta|\psi|^2$')
    cbar.set_ticks(np.round(np.linspace(-vabs, vabs, 3), 6))
    # Set colorbar ticks to scientific notation
    formatter = ScalarFormatter(useMathText=True)
    formatter.set_powerlimits((0, 0))
    formatter.set_useOffset(False)
    cbar.ax.yaxis.set_major_formatter(formatter)
    cbar.ax.yaxis.set_label_position('left')

    # set x-label
    for i, ax in enumerate([ax3, ax6, ax8]):
        ax.set_xlabel(r'$t$')

    # set y-label
    for i, ax in enumerate([ax1, ax7]):
        ax.set_ylabel(r'$x$')
    for i, ax in enumerate([ax2, ax3]):
        ax.set_ylabel(r'$k$')
    ax8.set_ylabel(r'$|\delta E| (\times 10^{-6})$')

    # set y-ticks dymamics
    for i, ax in enumerate([ax1, ax4, ax7]):
        ax.set_ylim([lim * 3 for lim in [-1.25, 1.25]])
        ax.set_yticks(np.linspace(-3, 3, 3))
    ax8.set_yticks(np.round(np.linspace(min(energy), max(energy),6),1))

    # set y-ticks params
    for i, ax in enumerate([ax2, ax3, ax5, ax6]):
        ax.set_yticks(np.round(np.linspace(0, num_params-1, 3)))

    # set imaginary x-ticks
    for i, ax in enumerate([ax1, ax2, ax3]):
        ax.set_xticks(np.round(np.linspace(0,max(t_imag),6),1))

    # set real x-ticks
    for i, ax in enumerate([ax4, ax5, ax6, ax7, ax8]):
        ax.set_xticks(np.round(np.linspace(0,max(t_real),6),1))
    ax8.sharex(ax7)
    # remove x-label ticks
    for i, ax in enumerate([ax1, ax2, ax4, ax5, ax7]):
        ax.tick_params(labelbottom=False)

    # remove y-label ticks
    for i, ax in enumerate([ax4, ax5, ax6]):
        ax.tick_params(labelleft=False)


    # Labeling
    abc = string.ascii_lowercase
    dx, dy = 3, 3  # distance in points from lower-left corner
    for i, ax in enumerate([ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8]):
        # create an offset transform from axes coordinates
        offset = mtransforms.ScaledTranslation(dx/72, dy/72, ax.figure.dpi_scale_trans)  # points to inches
        if i == 0 or i == 3:
            ax.text(0, 0, r"${\rm (%s)}$" % abc[i],
                horizontalalignment='left',
                verticalalignment='bottom',
                transform=ax.transAxes + offset,
                fontsize=15,
                color='white')
        else:
            ax.text(0, 0, r"${\rm (%s)}$" % abc[i],
                horizontalalignment='left',
                verticalalignment='bottom',
                transform=ax.transAxes + offset,
                fontsize=15)
            
        ax.tick_params(top=True, bottom=True, left=True, right=True,
                       direction='out')

    # plt.show()

    # Ensure figs directory exists
    if not os.path.exists(pm.figs_dir):
        os.makedirs(pm.figs_dir)
    # Save the figure
    fig.savefig(fig_path, format='pdf', bbox_inches='tight', transparent=False)

#--------------------------------
def evo_fig_breathing(mesh, t_real, den_real, params_real, \
                      energy, fig_path='fig_all.png'):
    
    
    # Enable LaTeX labels in Matplotlib
    plt.rcParams['text.usetex'] = True
    plt.rcParams['font.size'] = 17.5  # Optional: set the font size for clarity

    # Build the layout
    fig = plt.figure(constrained_layout=True, figsize=(8, 3))
    gs = gridspec.GridSpec(nrows=2, ncols=2, figure=fig)

    # First and second columns
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[0:2, 1])

    # Create a diverging colormap centered at zero
    cmap = 'RdBu' #'coolwarm'
    vabs = np.max([np.max(np.abs(params_real))])
    norm = TwoSlopeNorm(vmin=-vabs, vcenter=0, vmax=vabs)
    
    # Real evolution
    # define variables real
    real_params = np.real(params_real)
    num_params = params_real.shape[1]

    # Top Panel
    pcm_0 = ax1.pcolor(t_real, mesh, den_real, 
                            cmap='viridis', 
                            shading='auto')
    ax1.plot(t_real,np.cos(0.2025*t_real+1.5),'red')
    ax1.set_ylabel(r'$x$')

    # Middle Panel
    pcm_1 = ax2.pcolor(t_real, 
                            np.arange(num_params) if num_params > 1 else np.arange(num_params + 1), 
                            real_params.T if num_params > 1 else np.hstack([real_params, real_params]).T,
                            cmap=cmap, 
                            norm=norm, 
                            shading='auto')

    # Right Panel
    ax3.plot(t_real, energy)

    # Add colorbar to the side without affecting panel width
    cbar_0 = fig.colorbar(pcm_0, ax=ax1, pad=0, aspect=10, orientation='vertical')
    cbar_0.set_label(r'$|\psi(x,t)|^2$')
    cbar_0.set_ticks(np.round(np.linspace(0,max(map(max, den_real)),3),2))
    cbar_0.ax.yaxis.set_label_position('left')

    # Add colorbar to the side without affecting panel width
    cbar_1 = fig.colorbar(pcm_1, ax=[ax2],
                            pad=0,
                            aspect=10,
                            orientation='vertical')
    cbar_1.ax.yaxis.set_label_position('left')
    cbar_1.set_ticks(np.round(np.linspace(-vabs*0.85, vabs*0.85, 3), 2))
    cbar_1.set_label(r'${\rm Re}[\theta_k]$')


    # set x-label
    for i, ax in enumerate([ax2, ax3]):
        ax.set_xlabel(r'$t$')

    # set y-label
    for i, ax in enumerate([ax1]):
        ax.set_ylabel(r'$x$')
    for i, ax in enumerate([ax2]):
        ax.set_ylabel(r'$k$')
    ax3.set_ylabel(r'$|\delta E| (\times 10^{-3})$')

    # set y-ticks dymamics
    for i, ax in enumerate([ax1]):
        ax.set_ylim([lim * 3 for lim in [-1.25, 1.25]])
        ax.set_yticks(np.linspace(-3, 3, 3))
    ax3.set_yticks(np.round(np.linspace(min(energy), max(energy),6),1))

    # set y-ticks params
    for i, ax in enumerate([ax2]):
        ax.set_yticks(np.round(np.linspace(0, num_params-1, 3)))

    # set real x-ticks
    for i, ax in enumerate([ax1, ax2, ax3]):
        ax.set_xticks(np.round(np.linspace(0,max(t_real),6),1))
    # remove x-label ticks
    for i, ax in enumerate([ax1]):
        ax.tick_params(labelbottom=False)

    # Labeling
    abc = string.ascii_lowercase
    dx, dy = 3, 3  # distance in points from lower-left corner
    for i, ax in enumerate([ax1, ax2, ax3]):
        # create an offset transform from axes coordinates
        offset = mtransforms.ScaledTranslation(dx/72, dy/72, ax.figure.dpi_scale_trans)  # points to inches
        if i == 0 or i == 3:
            ax.text(0, 0, r"${\rm (%s)}$" % abc[i],
                horizontalalignment='left',
                verticalalignment='bottom',
                transform=ax.transAxes + offset,
                fontsize=15,
                color='white')
        else:
            ax.text(0, 0, r"${\rm (%s)}$" % abc[i],
                horizontalalignment='left',
                verticalalignment='bottom',
                transform=ax.transAxes + offset,
                fontsize=15)
            
        ax.tick_params(top=True, bottom=True, left=True, right=True,
                       direction='out')

    # plt.show()

    # Ensure figs directory exists
    if not os.path.exists(pm.figs_dir):
        os.makedirs(pm.figs_dir)
    # Save the figure
    fig.savefig(fig_path, format='pdf', bbox_inches='tight', transparent=False)

# --------------------------------
def evo_fig_soliton(mesh, t_real, \
                     den_real, \
                     params_real, \
                    den_diff, energy, fig_path='fig_all.png'):
    
    
    # Enable LaTeX labels in Matplotlib
    plt.rcParams['text.usetex'] = True
    plt.rcParams['font.size'] = 17.5  # Optional: set the font size for clarity

    # Build the layout
    fig = plt.figure(constrained_layout=True, figsize=(8, 3))
    gs = gridspec.GridSpec(nrows=2, ncols=2, figure=fig)

    # First and second columns
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[0, 1])
    ax4 = fig.add_subplot(gs[1, 1])

    
    # Create a diverging colormap centered at zero
    cmap = 'RdBu' #'coolwarm'
    vabs = np.max([np.max(np.abs(params_real))])
    norm = TwoSlopeNorm(vmin=-vabs, vcenter=0, vmax=vabs)
    
    # Real evolution
    # define variables real
    real_params = np.real(params_real)
    imag_params = np.imag(params_real)
    num_params = params_real.shape[1]

    # Top Panel
    pcm_0 = ax1.pcolor(t_real, mesh, den_real, 
                            cmap='viridis', 
                            shading='auto')
    ax1.set_ylabel(r'$x$')

    # Bottom Panel
    pcm_1 = ax2.pcolor(t_real, 
                            np.arange(num_params) if num_params > 1 else np.arange(num_params + 1), 
                            real_params.T if num_params > 1 else np.hstack([real_params, real_params]).T,
                            cmap=cmap, 
                            norm=norm, 
                            shading='auto')

    # Add colorbar to the side without affecting panel width
    cbar_0 = fig.colorbar(pcm_0, ax=ax1, pad=0, aspect=10, orientation='vertical')
    cbar_0.set_label(r'$|\psi(x,t)|^2$')
    cbar_0.set_ticks(np.round(np.linspace(0,max(map(max, den_real)),3),2))
    cbar_0.ax.yaxis.set_label_position('left')

    # Add colorbar to the side without affecting panel width
    cbar_1 = fig.colorbar(pcm_1, ax=[ax2],
                            pad=0,
                            aspect=10,
                            orientation='vertical')
    cbar_1.ax.yaxis.set_label_position('left')
    cbar_1.set_ticks(np.round(np.linspace(-vabs*0.85, vabs*0.85, 3), 2))
    cbar_1.set_label(r'${\rm Re}[\theta_k]$')

    
    # Comparison    
    cmap = 'RdBu' #'coolwarm'
    vabs = np.max(np.abs(den_diff))
    norm = TwoSlopeNorm(vmin=-vabs, vcenter=0, vmax=vabs)
    # Top panel
    pcm_2 = ax3.pcolor(t_real, mesh, den_diff, 
                      cmap=cmap, norm=norm, shading='auto')
    # Bottom panel
    ax4.plot(t_real, energy)

    # Add colorbar to the side without affecting panel width    
    cbar = fig.colorbar(pcm_2, ax=ax3, pad=0.01, aspect=10, orientation='vertical')
    cbar.set_label(r'$\delta|\psi|^2$')
    cbar.set_ticks(np.round(np.linspace(-vabs, vabs, 3), 6))
    # Set colorbar ticks to scientific notation
    formatter = ScalarFormatter(useMathText=True)
    formatter.set_powerlimits((0, 0))
    formatter.set_useOffset(False)
    cbar.ax.yaxis.set_major_formatter(formatter)
    cbar.ax.yaxis.set_label_position('left')

    # set x-label
    for i, ax in enumerate([ax2, ax4]):
        ax.set_xlabel(r'$t$')

    # set y-label
    for i, ax in enumerate([ax1, ax3]):
        ax.set_ylabel(r'$x$')
    for i, ax in enumerate([ax2]):
        ax.set_ylabel(r'$k$')
    ax4.set_ylabel(r'$|\delta E| (\times 10^{-4})$')

    # set y-ticks dymamics
    for i, ax in enumerate([ax1, ax3]):
        ax.set_ylim([lim * 3 for lim in [-1.25, 1.25]])
        ax.set_yticks(np.linspace(-3, 3, 3))
    ax4.set_yticks(np.round(np.linspace(min(energy), max(energy),3),1))

    # set y-ticks params
    for i, ax in enumerate([ax2]):
        ax.set_yticks(np.round(np.linspace(0, num_params-1, 3)))

    # set real x-ticks
    for i, ax in enumerate([ax1, ax2, ax3, ax4]):
        ax.set_xticks(np.round(np.linspace(0,max(t_real),6),1))
    ax4.sharex(ax3)
    # remove x-label ticks
    for i, ax in enumerate([ax1, ax3]):
        ax.tick_params(labelbottom=False)


    # Labeling
    abc = string.ascii_lowercase
    dx, dy = 3, 3  # distance in points from lower-left corner
    for i, ax in enumerate([ax1, ax2, ax3, ax4]):
        # create an offset transform from axes coordinates
        offset = mtransforms.ScaledTranslation(dx/72, dy/72, ax.figure.dpi_scale_trans)  # points to inches
        if i == 0 or i == 3:
            ax.text(0, 0, r"${\rm (%s)}$" % abc[i],
                horizontalalignment='left',
                verticalalignment='bottom',
                transform=ax.transAxes + offset,
                fontsize=15,
                color='white')
        else:
            ax.text(0, 0, r"${\rm (%s)}$" % abc[i],
                horizontalalignment='left',
                verticalalignment='bottom',
                transform=ax.transAxes + offset,
                fontsize=15)
            
        ax.tick_params(top=True, bottom=True, left=True, right=True,
                       direction='out')

    # plt.show()

    # Ensure figs directory exists
    if not os.path.exists(pm.figs_dir):
        os.makedirs(pm.figs_dir)
    # Save the figure
    fig.savefig(fig_path, format='pdf', bbox_inches='tight', transparent=False)

def evo_fig_complex(mesh, time, den, params, energy, fig_path='complex.png'):

    # Enable LaTeX labels in Matplotlib
    plt.rcParams['text.usetex'] = True
    plt.rcParams['font.size'] = 12  # Optional: set the font size for clarity
    
    # define variables
    t_max = max(time)
    real_params = np.real(params)
    imag_params = np.imag(params)
    num_params = params.shape[1]

    x_min = -8 # mesh[0] * 0.75  
    x_max = 8 # mesh[-1] * 0.75

    # Panel label
    ha = 'left'
    va = 'bottom'
    tx = 0.05
    ty = 0.05

    # Build the layout
    fig = plt.figure(constrained_layout=True, figsize=(6, 2))
    gs = gridspec.GridSpec(nrows=2, ncols=2, figure=fig, wspace=0.05, hspace=0)

    # First and second columns
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)
    ax3 = fig.add_subplot(gs[0, 1])
    ax4 = fig.add_subplot(gs[1, 1])

    # Top Left Panel
    pcm_0 = ax1.pcolor(time, mesh, den, 
                            cmap='viridis', 
                            shading='auto')
    # ax1.plot(time,-2*np.cos(0.2025*time+1.5*0),'red',linewidth=2/3)
    # ax1.plot(time,-2*np.cos(0.377*time+1.5*0),'red',linewidth=2/3)
    ax1.plot(time,-1*np.cos(0.1*time+1.5*0)+1,'red',linewidth=2/3)
    ax1.set_ylabel(r'$x/a_0$')
    ax1.tick_params(labelbottom=False, length=5)
    ax1.set_xticks(np.linspace(0, t_max, 6))
    ax1.set_ylim([lim for lim in [x_min, x_max]])
    ax1.set_yticks(np.linspace(x_min, x_max, 3))
    ax1.text(tx, ty, r"${\rm (a)}$",
                horizontalalignment=ha,
                verticalalignment=va,
                transform=ax1.transAxes,
                fontsize=12,
                color='white')

    # Add colorbar to the side without affecting panel width
    cbar_0 = fig.colorbar(pcm_0, ax=ax1, pad=0.0, aspect=10, orientation='vertical')  # Increased pad for spacing
    cbar_0.set_label(r'$|\psi(x,t)|^2$')
    cbar_0.set_ticks(np.round(np.linspace(0,max(map(max, den)),4),2))
    cbar_0.ax.yaxis.set_label_position('left')

    # Bottom Left Panel
    ax2.plot(time, energy)
    ax2.set_xlabel(r'$t/\tau$')
    ax2.set_ylabel(r'$\delta E/E_0$')
    ax2.set_xticks(np.linspace(0, t_max, 6))
    ax2.text(tx, ty, r"${\rm (b)}$",
                horizontalalignment=ha,
                verticalalignment=va,
                transform=ax2.transAxes,
                fontsize=12,
                color='black')
    ax2.text(0.95, 0.95, r'$(\times 10^{-4})$', transform=ax2.transAxes, 
             horizontalalignment='right', verticalalignment='top', fontsize=10)

    # Create a diverging colormap centered at zero
    cmap = 'RdBu' #'coolwarm'
    vabs = np.max([np.max(np.abs(real_params)), np.max(np.abs(imag_params))])
    norm = TwoSlopeNorm(vmin=-vabs, vcenter=0, vmax=vabs)

    # Top Right Panel
    pcm_1 = ax3.pcolor(time, 
                          np.arange(num_params) if num_params > 1 else np.arange(num_params + 1), 
                          real_params.T if num_params > 1 else np.hstack([real_params, real_params]).T,
                          cmap=cmap, 
                          norm=norm, 
                          shading='auto')
    ax3.set_ylabel(r'$k$')
    ax3.set_yticks(np.round(np.linspace(0, num_params-1, 3)))
    ax3.tick_params(labelbottom=False, length=5)
    ax3.set_xticks(np.round(np.linspace(0,max(time),6),1))
    ax3.text(tx,ty, r"${\rm (c)}$",
                horizontalalignment=ha,
                verticalalignment=va,
                transform=ax3.transAxes,
                fontsize=12)
    
    # Bottom Right Panel
    ax4.pcolor(time, 
                  np.arange(num_params) if num_params > 1 else np.arange(num_params + 1), 
                  imag_params.T if num_params > 1 else np.hstack([imag_params, imag_params]).T,
                  cmap=cmap, 
                  norm=norm, 
                  shading='auto')
    ax4.set_xlabel(r'$t/\tau$')
    ax4.set_ylabel(r'$k$')
    ax4.tick_params(labelbottom=True, length=5)
    ax4.set_xticks(np.round(np.linspace(0,max(time),6),1))
    ax4.set_yticks(np.round(np.linspace(0, num_params-1, 3)))
    ax4.text(tx, ty, r"${\rm (d)}$",
                horizontalalignment=ha,
                verticalalignment=va,
                transform=ax4.transAxes,
                fontsize=12)
    
    # Add colorbar to the side without affecting panel width
    cbar_1 = fig.colorbar(pcm_1, ax=[ax3, ax4],
                          pad=0.0,
                          aspect=20,
                          orientation='vertical')
    cbar_1.ax.yaxis.set_label_position('left')
    cbar_1.set_ticks(np.round(np.linspace(-vabs*0.85, vabs*0.85, 5), 2))
    cbar_1.set_label(r'${\rm Im}[\delta\theta_k] \hspace{2em} {\rm Re}[\delta\theta_k]$')

    

    # Adjust vertical spacing
    fig.get_layout_engine().set(w_pad=0, h_pad=0, hspace=0, wspace=0)

    # plt.show()

    # Ensure figs directory exists
    if not os.path.exists(pm.figs_dir):
        os.makedirs(pm.figs_dir)
    # Save the figure
    fig.savefig(fig_path, format=pm.fig_format, bbox_inches='tight', transparent=False)
def energy_fig_with_gs(t_dyn, energy_dyn, energy_gs=None, fig_path='energy.pdf',
                       ylabel=r'$E(t)$', relative=False):
    plt.rcParams['text.usetex'] = True
    plt.rcParams['font.size'] = 12

    if energy_gs is not None:
        time = np.concatenate(([0], np.asarray(t_dyn)))
        energy = np.concatenate(([energy_gs], np.asarray(energy_dyn)))
    else:
        time = np.asarray(t_dyn)
        energy = np.asarray(energy_dyn)

    if relative:
        energy = (energy - energy[1]) / energy[1]

    fig, ax = plt.subplots(figsize=(4, 2.5), layout='constrained')

    ax.plot(time, energy, marker='o', markersize=2, linewidth=1.2)
    ax.set_xlabel(r'$t/\tau$')
    ax.set_ylabel(ylabel)
    ax.set_xticks(np.round(np.linspace(0, max(time), 6), 1))

    e_min = np.min(energy)
    e_max = np.max(energy)
    if np.isclose(e_min, e_max):
        ax.set_yticks([e_min])
    else:
        ax.set_yticks(np.linspace(e_min, e_max, 4))

    if not os.path.exists(pm.figs_dir):
        os.makedirs(pm.figs_dir)

    fig.savefig(fig_path, format=pm.fig_format, bbox_inches='tight', transparent=False)
def ho_benchmark_density(mesh, rho_num, rho_exact, w, fig_path='ho_benchmark_density.pdf'):
    plt.rcParams['text.usetex'] = True
    plt.rcParams['font.size'] = 12

    fig, ax = plt.subplots(figsize=(4, 2.5), layout='constrained')

    ax.plot(mesh, rho_num, label='NQS')
    ax.plot(mesh, rho_exact, '--', linewidth=2, label=fr'Exact HO: $\omega={w}$')

    x_max = mesh[np.argmax(rho_num)]
    ax.vlines(x_max, 0, rho_num.max(), linestyles=':', alpha=0.5)

    ax.set_xlabel(r'$x/a_0$')
    ax.set_ylabel(r'$|\psi(x)|^2$')
    ax.legend()

    if not os.path.exists(pm.figs_dir):
        os.makedirs(pm.figs_dir)

    fig.savefig(fig_path, format=pm.fig_format, bbox_inches='tight', transparent=False)
def ho_kick_benchmark_fig(t, x_mean_num, x_mean_exact, fig_path='ho_kick_benchmark.pdf'):
    plt.rcParams['text.usetex'] = True
    plt.rcParams['font.size'] = 12

    fig, ax = plt.subplots(figsize=(4, 2.5), layout='constrained')

    ax.plot(t, x_mean_num, label='NQS')
    ax.plot(t, x_mean_exact, '--', linewidth=2, label='Exact HO')

    ax.set_xlabel(r'$t/\tau$')
    ax.set_ylabel(r'$\langle x\rangle(t)$')
    ax.set_xticks(np.round(np.linspace(0, max(t), 6), 1))
    ax.legend()

    if not os.path.exists(pm.figs_dir):
        os.makedirs(pm.figs_dir)

    fig.savefig(fig_path, format=pm.fig_format, bbox_inches='tight', transparent=False)
def ho_quench_benchmark_fig(t, var_num, var_exact, fig_path='ho_quench_benchmark.pdf'):
    plt.rcParams['text.usetex'] = True
    plt.rcParams['font.size'] = 12

    fig, ax = plt.subplots(figsize=(4, 2.5), layout='constrained')

    ax.plot(t, var_num, label='NQS')
    ax.plot(t, var_exact, '--', linewidth=2, label='Exact HO quench')

    ax.set_xlabel(r'$t/\tau$')
    ax.set_ylabel(r'$\sigma_x^2(t)$')
    ax.set_xticks(np.round(np.linspace(0, max(t), 6), 1))
    ax.legend()

    if not os.path.exists(pm.figs_dir):
        os.makedirs(pm.figs_dir)

    fig.savefig(fig_path, format=pm.fig_format, bbox_inches='tight', transparent=False)
def ho_quench_frequency_fig(t, var_num, var_exact, peak_times, fig_path='ho_quench_frequency.pdf'):
    plt.rcParams['text.usetex'] = True
    plt.rcParams['font.size'] = 12

    fig, ax = plt.subplots(figsize=(4, 2.5), layout='constrained')

    ax.plot(t, var_num, label='NQS')
    ax.plot(t, var_exact, '--', linewidth=2, label='Exact HO quench')

    if len(peak_times) > 0:
        peak_values = np.interp(peak_times, t, var_num)
        ax.plot(peak_times, peak_values, 'o', markersize=4, label='Detected peaks')

    ax.set_xlabel(r'$t/\tau$')
    ax.set_ylabel(r'$\sigma_x^2(t)$')
    ax.set_xticks(np.round(np.linspace(0, max(t), 6), 1))
    ax.legend()

    if not os.path.exists(pm.figs_dir):
        os.makedirs(pm.figs_dir)

    fig.savefig(fig_path, format=pm.fig_format, bbox_inches='tight', transparent=False)
def bright_soliton_benchmark_density(mesh, rho_num, rho_exact, rho_max, fig_path='bright_soliton_benchmark.pdf'):
    plt.rcParams['text.usetex'] = True
    plt.rcParams['font.size'] = 12

    fig, ax = plt.subplots(figsize=(4, 2.5), layout='constrained')

    ax.plot(mesh, rho_num, label='NQS')
    ax.plot(
        mesh,
        rho_exact,
        '--',
        linewidth=2,
        label=rf'Exact soliton: $\rho_{{\max}}={rho_max:.3f}$'
    )

    x_max = mesh[np.argmax(rho_num)]
    ax.vlines(x_max, 0, rho_num.max(), linestyles=':', alpha=0.5)

    ax.set_xlabel(r'$x/a_0$')
    ax.set_ylabel(r'$|\psi(x)|^2$')
    ax.legend()

    if not os.path.exists(pm.figs_dir):
        os.makedirs(pm.figs_dir)

    fig.savefig(fig_path, format=pm.fig_format, bbox_inches='tight', transparent=False)
def bright_soliton_kick_benchmark_fig(
    t,
    x_max_num,
    x_max_exact,
    rho_max_t,
    norm_t,
    fig_path='bright_soliton_kick_benchmark.pdf'
):
    plt.rcParams['text.usetex'] = True
    plt.rcParams['font.size'] = 12

    fig, axes = plt.subplots(3, 1, figsize=(4.5, 5.5), sharex=True, layout='constrained')

    # Top panel: position of the density maximum
    axes[0].plot(t, x_max_num, label='NQS')
    axes[0].plot(t, x_max_exact, '--', linewidth=2, label='Expected')
    axes[0].set_ylabel(r'$x_{\max}(t)$')
    axes[0].legend()

    # Middle panel: peak density
    axes[1].plot(t, rho_max_t, label='NQS')
    axes[1].axhline(rho_max_t[0], linestyle='--', linewidth=1.5, label='Initial')
    axes[1].set_ylabel(r'$\rho_{\max}(t)$')
    axes[1].legend()

    # Bottom panel: norm
    axes[2].plot(t, norm_t, label='NQS')
    axes[2].axhline(norm_t[0], linestyle='--', linewidth=1.5, label='Initial')
    axes[2].set_ylabel(r'$N(t)$')
    axes[2].set_xlabel(r'$t/\tau$')
    axes[2].legend()

    if not os.path.exists(pm.figs_dir):
        os.makedirs(pm.figs_dir)

    fig.savefig(fig_path, format=pm.fig_format, bbox_inches='tight', transparent=False)
def energy_imag_fig(time, energy, fig_path='energy_imag.pdf', title=None):
    plt.rcParams['text.usetex'] = True
    plt.rcParams['font.size'] = 12

    time = np.asarray(time).squeeze()
    energy = np.real(np.asarray(energy).squeeze())

    E_final = energy[-1]
    rel_to_final = (energy - E_final) / np.abs(E_final) if not np.isclose(E_final, 0.0) else energy - E_final

    fig, axs = plt.subplots(2, 1, figsize=(4.2, 4.0), sharex=True, layout='constrained')

    axs[0].plot(time, energy, lw=1.5)
    axs[0].axhline(E_final, linestyle='--', linewidth=1)
    axs[0].set_ylabel(r'$E(\tau)$')
    axs[0].grid(alpha=0.25)

    axs[1].plot(time, rel_to_final, lw=1.5)
    axs[1].set_ylabel(r'$\frac{E(\tau)-E_f}{|E_f|}$')
    axs[1].set_xlabel(r'$\tau$')
    axs[1].grid(alpha=0.25)

    if title is not None:
        fig.suptitle(title)

    if not os.path.exists(pm.figs_dir):
        os.makedirs(pm.figs_dir)

    fig.savefig(fig_path, format=pm.fig_format, bbox_inches='tight', transparent=False)
def bright_soliton_summary_fig(
    mesh,
    time,
    den,
    energy,
    rho_final,
    rho_exact,
    fig_path='bright_soliton_summary.pdf',
    title=None
):
    plt.rcParams['text.usetex'] = True
    plt.rcParams['font.size'] = 12

    mesh = np.asarray(mesh)
    time = np.asarray(time)
    den = np.asarray(den)
    energy = np.real(np.asarray(energy).squeeze())
    rho_final = np.asarray(rho_final)
    rho_exact = np.asarray(rho_exact)

    fig, axs = plt.subplots(1, 3, figsize=(15, 4.2), layout='constrained')
    # (a) Density evolution
    pcm = axs[0].pcolormesh(
        time,
        mesh,
        den.T,
        cmap='viridis',
        shading='auto'
    )
    axs[0].set_ylabel(r'$x/a_0$')
    axs[0].set_title(r'(a) Density evolution')
    cbar = fig.colorbar(pcm, ax=axs[0], pad=0.01)
    cbar.set_label(r'$|\psi(x,t)|^2$')

    # (b) Final density vs sech profile
    axs[1].plot(mesh, rho_final, label='Final NQS density')
    axs[1].plot(mesh, rho_exact, '--', linewidth=2, label='Analytical')
    x_max = mesh[np.argmax(rho_final)]
    axs[1].axvline(x_max, linestyle=':', alpha=0.5, color='k')
    axs[1].set_ylabel(r'$|\psi(x)|^2$')
    axs[1].set_title(r'(b) Final density vs sech profile')
    axs[1].legend()
    axs[1].grid(alpha=0.25)

    # (c) Energy evolution
    axs[2].plot(time, energy, linewidth=1.5)
    axs[2].axhline(energy[-1], linestyle='--', linewidth=1, alpha=0.7)
    axs[2].set_xlabel(r'$t/\tau$')
    axs[2].set_ylabel(r'$E(t)$')
    axs[2].set_title(r'(c) Energy evolution')
    axs[2].grid(alpha=0.25)

    if title is not None:
        fig.suptitle(title)

    if not os.path.exists(pm.figs_dir):
        os.makedirs(pm.figs_dir)

    fig.savefig(fig_path, format=pm.fig_format, bbox_inches='tight', transparent=False)
    

def bright_soliton_kick_summary_2panel(
    t,
    x,
    rho_xt,
    x_max_num,
    x_max_exact,
    fig_path=None,
    title=None
):
    """
    Summary figure for the bright-soliton kick benchmark with 2 vertical subplots:
      1) density evolution
      2) peak trajectory x_max(t): numerical vs analytical
    """

    fig, axes = plt.subplots(2, 1, figsize=(6, 8), sharex=True)

    # --------------------------------------------------
    # 1) Density evolution
    # --------------------------------------------------
    im = axes[0].pcolormesh(
        t,
        x,
        rho_xt.T,
        shading='auto'
    )
    axes[0].set_ylabel(r'$x$')
    axes[0].set_title(r'Density evolution: $|\psi(x,t)|^2$')
    plt.colorbar(im, ax=axes[0], label=r'$|\psi|^2$')

    # --------------------------------------------------
    # 2) Peak trajectory
    # --------------------------------------------------
    axes[1].plot(t, x_max_num, label='NQS')
    axes[1].plot(t, x_max_exact, '--', label='Analytical')
    axes[1].set_xlabel(r'$t$')
    axes[1].set_ylabel(r'$x_{\max}(t)$')
    axes[1].set_title(r'Peak trajectory')
    axes[1].grid(alpha=0.2)
    axes[1].legend()

    if title is not None:
        fig.suptitle(title)

    plt.tight_layout()

    if fig_path is not None:
        folder = os.path.dirname(fig_path)
        if folder and not os.path.exists(folder):
            os.makedirs(folder, exist_ok=True)
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')

    plt.show()


def bright_soliton_kick_summary_3panel(
    t,
    x,
    rho_xt,
    x_max_num,
    x_max_exact,
    rho_max_num,
    rho_max_exact=None,
    fig_path=None,
    title=None
):
    """
    Summary figure for the bright-soliton kick benchmark with 3 vertical subplots:
      1) density evolution
      2) peak trajectory x_max(t): numerical vs analytical
      3) peak height rho_max(t)
    """

    fig, axes = plt.subplots(3, 1, figsize=(6, 10), sharex=True)

    # --------------------------------------------------
    # 1) Density evolution
    # --------------------------------------------------
    im = axes[0].pcolormesh(
        t,
        x,
        rho_xt.T,
        shading='auto'
    )
    axes[0].set_ylabel(r'$x$')
    axes[0].set_title(r'Density evolution: $|\psi(x,t)|^2$')
    plt.colorbar(im, ax=axes[0], label=r'$|\psi|^2$')

    # --------------------------------------------------
    # 2) Peak trajectory
    # --------------------------------------------------
    axes[1].plot(t, x_max_num, label='NQS')
    axes[1].plot(t, x_max_exact, '--', label='Analytical')
    axes[1].set_ylabel(r'$x_{\max}(t)$')
    axes[1].set_title(r'Peak trajectory')
    axes[1].grid(alpha=0.2)
    axes[1].legend()

    # --------------------------------------------------
    # 3) Peak height
    # --------------------------------------------------
    axes[2].plot(t, rho_max_num, label='NQS')

    if rho_max_exact is None:
        rho_max_exact = np.full_like(t, rho_max_num[0], dtype=float)

    axes[2].plot(t, rho_max_exact, '--', label='Analytical')
    axes[2].set_xlabel(r'$t$')
    axes[2].set_ylabel(r'$\rho_{\max}(t)$')
    axes[2].set_title(r'Peak height evolution')
    axes[2].grid(alpha=0.2)
    axes[2].legend()

    if title is not None:
        fig.suptitle(title)

    plt.tight_layout()

    if fig_path is not None:
        folder = os.path.dirname(fig_path)
        if folder and not os.path.exists(folder):
            os.makedirs(folder, exist_ok=True)
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')

    plt.show()
def ho_quench_summary_fig(
    mesh,
    rho_gs_num,
    rho_gs_exact,
    time,
    den,
    var_num,
    var_exact,
    fig_path='ho_quench_summary.pdf',
    title=None
):
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    import parameters as pm

    plt.rcParams['text.usetex'] = True
    plt.rcParams['font.size'] = 12

    mesh = np.asarray(mesh)
    rho_gs_num = np.asarray(rho_gs_num)
    rho_gs_exact = np.asarray(rho_gs_exact)
    time = np.asarray(time)
    den = np.asarray(den)
    var_num = np.asarray(var_num)
    var_exact = np.asarray(var_exact)

    fig, axs = plt.subplots(
        1, 3,
        figsize=(14.5, 3.8),
        layout='constrained'
    )

    # --------------------------------------------------
    # (a) Ground state from imaginary-time evolution
    # --------------------------------------------------
    axs[0].plot(mesh, rho_gs_num, label='NQS GS')
    axs[0].plot(mesh, rho_gs_exact, '--', linewidth=2, label='Exact HO GS')
    axs[0].set_xlabel(r'$x/a_0$')
    axs[0].set_ylabel(r'$|\psi(x)|^2$')
    axs[0].set_title(r'(a) Ground state')
    axs[0].grid(alpha=0.25)
    axs[0].legend()

    # --------------------------------------------------
    # (b) Density evolution during the real-time dynamics
    # --------------------------------------------------
    pcm = axs[1].pcolormesh(
        time,
        mesh,
        den.T,
        cmap='viridis',
        shading='auto'
    )
    axs[1].set_xlabel(r'$t/\tau$')
    axs[1].set_ylabel(r'$x/a_0$')
    axs[1].set_title(r'(b) Density evolution')
    cbar = fig.colorbar(pcm, ax=axs[1], pad=0.01)
    cbar.set_label(r'$|\psi(x,t)|^2$')

    # --------------------------------------------------
    # (c) Width / variance comparison with the analytical result
    # --------------------------------------------------
    axs[2].plot(time, var_num, label='NQS')
    axs[2].plot(time, var_exact, '--', linewidth=2, label='Exact HO quench')
    axs[2].set_xlabel(r'$t/\tau$')
    axs[2].set_ylabel(r'$\sigma_x^2(t)$')
    axs[2].set_title(r'(c) Width comparison')
    axs[2].grid(alpha=0.25)
    axs[2].legend()

    if title is not None:
        fig.suptitle(title)

    if not os.path.exists(pm.figs_dir):
        os.makedirs(pm.figs_dir)

    fig.savefig(
        fig_path,
        format=pm.fig_format,
        bbox_inches='tight',
        transparent=False
    )
def bright_soliton_two_step_summary_from_lists(
    mesh_list,
    den_list,
    w_list,
    time_list,
    energy_list,
    rho_final,
    rho_exact,
    fig_path='bright_soliton_two_step_summary.pdf',
    title=None,
    xlim=(-10, 10)
):
    """
    Summary figure for two-step bright-soliton preparation using the lists
    already generated in the ground-state loop.

    The function expects:
      - one trapped imaginary-time evolution, omega != 0
      - one free imaginary-time evolution, omega = 0

    Layout:
      (a) density evolution in the trap
      (b) density evolution after removing the trap
      (c) final NQS density vs analytical profile
      (d) energy evolution during the free step
    """

    set_thesis_plot_style()

    # Convert to numpy
    mesh_list = [
        m.detach().cpu().numpy() if hasattr(m, "detach") else np.asarray(m)
        for m in mesh_list
    ]
    den_list = [np.asarray(d) for d in den_list]
    time_list = [np.asarray(t) for t in time_list]
    energy_list = [np.real(np.asarray(e).squeeze()) for e in energy_list]
    w_list = np.asarray(w_list)

    # Identify trapped and free evolutions
    trap_idx = np.where(~np.isclose(w_list, 0.0))[0][0]
    free_idx = np.where(np.isclose(w_list, 0.0))[0][0]

    mesh = mesh_list[free_idx]
    mask = (mesh >= xlim[0]) & (mesh <= xlim[1])
    mesh_plot = mesh[mask]

    time_trap = time_list[trap_idx]
    time_free = time_list[free_idx]

    den_trap = den_list[trap_idx]
    den_free = den_list[free_idx]

    energy_free = energy_list[free_idx]

    rho_final = np.asarray(rho_final)
    rho_exact = np.asarray(rho_exact)

    fig = plt.figure(figsize=(7.0, 2.25), constrained_layout=True)

    gs = gridspec.GridSpec(
        nrows=2,
        ncols=3,
        figure=fig,
        width_ratios=[0.95, 1.15, 1.15],
        height_ratios=[1.0, 1.0],
    )

    ax_trap = fig.add_subplot(gs[0, 0])
    ax_free = fig.add_subplot(gs[1, 0], sharex=ax_trap)
    ax_profile = fig.add_subplot(gs[:, 1])
    ax_energy = fig.add_subplot(gs[:, 2])

    # --------------------------------------------------
    # (a) First step: weak trap
    # --------------------------------------------------
    pcm_trap = ax_trap.pcolormesh(
        time_trap,
        mesh_plot,
        den_trap[:, mask].T,
        cmap='viridis',
        shading='auto'
    )

    ax_trap.set_title(r'(a) Trap, $\omega=0.1$')
    ax_trap.set_ylabel(r'$x/a_0$')
    ax_trap.set_ylim(xlim)
    ax_trap.tick_params(labelbottom=False)

    cbar_trap = fig.colorbar(pcm_trap, ax=ax_trap, pad=0.01)
    cbar_trap.set_label(r'$|\psi(x,\tau)|^2$')
    cbar_trap.ax.tick_params(labelsize=11)

    # --------------------------------------------------
    # (b) Second step: free soliton
    # --------------------------------------------------
    pcm_free = ax_free.pcolormesh(
        time_free,
        mesh_plot,
        den_free[:, mask].T,
        cmap='viridis',
        shading='auto'
    )

    ax_free.set_title(r'(b) Free, $\omega=0$')
    ax_free.set_xlabel(r'$\tau$')
    ax_free.set_ylabel(r'$x/a_0$')
    ax_free.set_ylim(xlim)

    cbar_free = fig.colorbar(pcm_free, ax=ax_free, pad=0.01)
    cbar_free.set_label(r'$|\psi(x,\tau)|^2$')
    cbar_free.ax.tick_params(labelsize=11)

    # Shared temporal axis for the two density maps
    t_max = max(np.max(time_trap), np.max(time_free))
    ax_free.set_xlim(0, t_max)
    ax_free.set_xticks(np.round(np.linspace(0, t_max, 6), 1))

    # --------------------------------------------------
    # (c) Final density profile
    # --------------------------------------------------
    ax_profile.plot(
        mesh_plot,
        rho_final[mask],
        linewidth=1.6,
        label='Final NQS density'
    )

    ax_profile.plot(
        mesh_plot,
        rho_exact[mask],
        '--',
        linewidth=1.8,
        label='Analytical sech profile'
    )

    x_max = mesh[np.argmax(rho_final)]
    ax_profile.axvline(
        x_max,
        linestyle=':',
        linewidth=1.0,
        alpha=0.6,
        color='k'
    )

    ax_profile.set_title(r'(c) Final density')
    ax_profile.set_xlabel(r'$x/a_0$')
    ax_profile.set_ylabel(r'$|\psi(x)|^2$')
    ax_profile.grid(alpha=0.25)
    ax_profile.legend()

    # --------------------------------------------------
    # (d) Energy evolution
    # --------------------------------------------------
    ax_energy.plot(
        time_free,
        energy_free,
        linewidth=1.6
    )

    ax_energy.axhline(
        energy_free[-1],
        linestyle='--',
        linewidth=1.0,
        alpha=0.7
    )

    ax_energy.set_title(r'(d) Energy evolution')
    ax_energy.set_xlabel(r'$\tau$')
    ax_energy.set_ylabel(r'$E(\tau)$')
    ax_energy.grid(alpha=0.25)

    if title is not None:
        fig.suptitle(title)

    if not os.path.exists(pm.figs_dir):
        os.makedirs(pm.figs_dir)

    fig.savefig(
        fig_path,
        format=pm.fig_format,
        bbox_inches='tight',
        transparent=False
    )

def bright_kick_density_params_side_by_side(
    time,
    mesh,
    den,
    params,
    x_ref=None,
    fig_path='bright_kick_density_params.pdf',
    title=None
):
    set_thesis_plot_style()

    time = np.asarray(time)
    mesh = np.asarray(mesh)
    den = np.asarray(den)
    params = np.asarray(params)

    # Ensure density has shape (Nt, Nx)
    if den.shape[0] != len(time) and den.shape[1] == len(time):
        den = den.T

    real_params = np.real(params)
    imag_params = np.imag(params)
    num_params = params.shape[1]

    vabs = np.max([
        np.max(np.abs(real_params)) if real_params.size else 0.0,
        np.max(np.abs(imag_params)) if imag_params.size else 0.0
    ])

    if np.isclose(vabs, 0.0):
        vabs = 1.0

    norm = TwoSlopeNorm(vmin=-vabs, vcenter=0.0, vmax=vabs)
    cmap_params = 'RdBu_r'

    fig = plt.figure(figsize=(10.5, 3.0), constrained_layout=True)

    outer = gridspec.GridSpec(
        1,
        2,
        figure=fig,
        width_ratios=[1.1, 1.0]
    )

    ax_den = fig.add_subplot(outer[0, 0])

    right = gridspec.GridSpecFromSubplotSpec(
        2,
        1,
        subplot_spec=outer[0, 1],
        hspace=0.05
    )

    ax_re = fig.add_subplot(right[0, 0])
    ax_im = fig.add_subplot(right[1, 0], sharex=ax_re)

    # --------------------------------------------------
    # Density evolution
    # --------------------------------------------------
    pcm_den = ax_den.pcolormesh(
        time,
        mesh,
        den.T,
        cmap='viridis',
        shading='auto'
    )

    if x_ref is not None:
        ax_den.plot(
            time,
            x_ref,
            color='red',
            linestyle='--',
            linewidth=1.5,
            label='Ballistic reference'
        )
        ax_den.legend(
            loc='upper left',
            frameon=True,
            fontsize=8
        )

    ax_den.set_title(r'(a) Density evolution')
    ax_den.set_xlabel(r'$t$')
    ax_den.set_ylabel(r'$x$')

    cbar_den = fig.colorbar(pcm_den, ax=ax_den, pad=0.01)
    cbar_den.set_label(r'$|\psi(x,t)|^2$', fontsize=10)
    cbar_den.ax.tick_params(labelsize=9)

    # --------------------------------------------------
    # Real and imaginary parts of parameters
    # --------------------------------------------------
    if num_params == 1:
        y_params = np.arange(2)
        re_plot = np.vstack([real_params[:, 0], real_params[:, 0]])
        im_plot = np.vstack([imag_params[:, 0], imag_params[:, 0]])
    else:
        y_params = np.arange(num_params)
        re_plot = real_params.T
        im_plot = imag_params.T

    pcm_re = ax_re.pcolormesh(
        time,
        y_params,
        re_plot,
        cmap=cmap_params,
        norm=norm,
        shading='auto'
    )

    ax_re.set_title(r'Parameter evolution')
    ax_re.set_ylabel(r'$j$')
    ax_re.tick_params(labelbottom=False)

    ax_re.text(
        0.03, 0.82, r'(b)',
        transform=ax_re.transAxes,
        fontsize=11,
        ha='left',
        va='top',
        bbox=dict(
            facecolor='white',
            edgecolor='none',
            alpha=0.7,
            pad=1.5
        )
    )

    ax_im.pcolormesh(
        time,
        y_params,
        im_plot,
        cmap=cmap_params,
        norm=norm,
        shading='auto'
    )

    ax_im.set_xlabel(r'$t$')
    ax_im.set_ylabel(r'$j$')

    ax_im.text(
        0.03, 0.82, r'(c)',
        transform=ax_im.transAxes,
        fontsize=11,
        ha='left',
        va='top',
        bbox=dict(
            facecolor='white',
            edgecolor='none',
            alpha=0.7,
            pad=1.5
        )
    )

    if num_params > 1:
        ticks = np.round(
            np.linspace(0, num_params - 1, min(4, num_params))
        ).astype(int)
        ax_re.set_yticks(ticks)
        ax_im.set_yticks(ticks)

    cbar_par = fig.colorbar(
        pcm_re,
        ax=[ax_re, ax_im],
        pad=0.01
    )

    cbar_par.set_label(
        r'${\rm Re}[\theta_j]\;/\;{\rm Im}[\theta_j]$',
        fontsize=10
    )
    cbar_par.ax.tick_params(labelsize=9)

    if title is not None:
        fig.suptitle(title)

    if not os.path.exists(pm.figs_dir):
        os.makedirs(pm.figs_dir)

    fig.savefig(
        fig_path,
        format=pm.fig_format,
        bbox_inches='tight',
        transparent=False
    )
def bright_kick_density_params_side_by_side2(
    time,
    mesh,
    den,
    params,
    x_ref=None,
    xlim=None,
    fig_path='bright_kick_density_params.pdf',
    title=None
):
    set_thesis_plot_style()

    time = np.asarray(time)
    mesh = np.asarray(mesh)
    den = np.asarray(den)
    params = np.asarray(params)

    # Ensure density has shape (Nt, Nx)
    if den.shape[0] != len(time) and den.shape[1] == len(time):
        den = den.T

    # --------------------------------------------------
    # Restrict spatial plotting window, e.g. xlim=(-15, 15)
    # --------------------------------------------------
    if xlim is not None:
        x_min, x_max = xlim
        x_mask = (mesh >= x_min) & (mesh <= x_max)

        mesh_plot = mesh[x_mask]
        den_plot = den[:, x_mask]
    else:
        mesh_plot = mesh
        den_plot = den

    if x_ref is not None:
        x_ref_plot = np.asarray(x_ref)
    else:
        x_ref_plot = None

    real_params = np.real(params)
    imag_params = np.imag(params)
    num_params = params.shape[1]

    vabs = np.max([
        np.max(np.abs(real_params)) if real_params.size else 0.0,
        np.max(np.abs(imag_params)) if imag_params.size else 0.0
    ])

    if np.isclose(vabs, 0.0):
        vabs = 1.0

    norm = TwoSlopeNorm(vmin=-vabs, vcenter=0.0, vmax=vabs)
    cmap_params = 'RdBu_r'

    fig = plt.figure(figsize=(10.5, 3.0), constrained_layout=True)

    outer = gridspec.GridSpec(
        1,
        2,
        figure=fig,
        width_ratios=[1.1, 1.0]
    )

    ax_den = fig.add_subplot(outer[0, 0])

    right = gridspec.GridSpecFromSubplotSpec(
        2,
        1,
        subplot_spec=outer[0, 1],
        hspace=0.05
    )

    ax_re = fig.add_subplot(right[0, 0])
    ax_im = fig.add_subplot(right[1, 0], sharex=ax_re)

    # --------------------------------------------------
    # Density evolution
    # --------------------------------------------------
    pcm_den = ax_den.pcolormesh(
        time,
        mesh_plot,
        den_plot.T,
        cmap='viridis',
        shading='auto'
    )

    # Plot ballistic reference only inside the displayed spatial window
    if x_ref_plot is not None:
        if xlim is not None:
            x_min, x_max = xlim
            ref_mask = (x_ref_plot >= x_min) & (x_ref_plot <= x_max)

            ax_den.plot(
                time[ref_mask],
                x_ref_plot[ref_mask],
                color='red',
                linestyle='--',
                linewidth=1.5,
                label='Ballistic reference'
            )
        else:
            ax_den.plot(
                time,
                x_ref_plot,
                color='red',
                linestyle='--',
                linewidth=1.5,
                label='Ballistic reference'
            )

        ax_den.legend(
            loc='upper left',
            frameon=True,
            fontsize=8
        )

    ax_den.set_title(r'(a) Density evolution')
    ax_den.set_xlabel(r'$t$')
    ax_den.set_ylabel(r'$x$')

    if xlim is not None:
        ax_den.set_ylim(xlim)

    cbar_den = fig.colorbar(pcm_den, ax=ax_den, pad=0.01)
    cbar_den.set_label(r'$|\psi(x,t)|^2$', fontsize=10)
    cbar_den.ax.tick_params(labelsize=9)

    # --------------------------------------------------
    # Real and imaginary parts of parameters
    # --------------------------------------------------
    if num_params == 1:
        y_params = np.arange(2)
        re_plot = np.vstack([real_params[:, 0], real_params[:, 0]])
        im_plot = np.vstack([imag_params[:, 0], imag_params[:, 0]])
    else:
        y_params = np.arange(num_params)
        re_plot = real_params.T
        im_plot = imag_params.T

    pcm_re = ax_re.pcolormesh(
        time,
        y_params,
        re_plot,
        cmap=cmap_params,
        norm=norm,
        shading='auto'
    )

    ax_re.set_title(r'Parameter evolution')
    ax_re.set_ylabel(r'$j$')
    ax_re.tick_params(labelbottom=False)

    ax_re.text(
        0.03, 0.82, r'(b)',
        transform=ax_re.transAxes,
        fontsize=11,
        ha='left',
        va='top',
        bbox=dict(
            facecolor='white',
            edgecolor='none',
            alpha=0.7,
            pad=1.5
        )
    )

    pcm_im = ax_im.pcolormesh(
        time,
        y_params,
        im_plot,
        cmap=cmap_params,
        norm=norm,
        shading='auto'
    )

    ax_im.set_xlabel(r'$t$')
    ax_im.set_ylabel(r'$j$')

    ax_im.text(
        0.03, 0.82, r'(c)',
        transform=ax_im.transAxes,
        fontsize=11,
        ha='left',
        va='top',
        bbox=dict(
            facecolor='white',
            edgecolor='none',
            alpha=0.7,
            pad=1.5
        )
    )

    if num_params > 1:
        ticks = np.round(
            np.linspace(0, num_params - 1, min(4, num_params))
        ).astype(int)
        ax_re.set_yticks(ticks)
        ax_im.set_yticks(ticks)

    cbar_par = fig.colorbar(
        pcm_re,
        ax=[ax_re, ax_im],
        pad=0.01
    )

    cbar_par.set_label(
        r'${\rm Re}[\theta_j]\;/\;{\rm Im}[\theta_j]$',
        fontsize=10
    )
    cbar_par.ax.tick_params(labelsize=9)

    if title is not None:
        fig.suptitle(title)

    if not os.path.exists(pm.figs_dir):
        os.makedirs(pm.figs_dir)

    fig.savefig(
        fig_path,
        format=pm.fig_format,
        bbox_inches='tight',
        transparent=False
    )