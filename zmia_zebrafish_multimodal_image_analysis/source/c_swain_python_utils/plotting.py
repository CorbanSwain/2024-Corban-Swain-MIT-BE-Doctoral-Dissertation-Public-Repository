#!python3
# plotting.py

# Project: c_swain_python_utils
# by Corban Swain 2019


import os
import time
import c_swain_python_utils as csutils


__all__ = ['despine', 'despine_all', 'save_figures', 'set_mpl_defaults',
           'stamp_fig']


def despine_all(ax):
    despine(ax, **{pos: True for pos in ('left', 'right', 'top', 'bottom')})


def despine(ax, **kwargs):
    [ax.spines[k].set_visible(not v) for k, v in kwargs.items()]


def stamp_fig(fig, stamp_str='figure %n | %d', **kwargs):
    import matplotlib.pyplot as plt

    replace_dict = {
        '%n': f'{fig.number:d}',
        '%d': time.strftime('%y%m%d-%H%M')}

    full_stamp_str = stamp_str

    for r in replace_dict.items():
        full_stamp_str = full_stamp_str.replace(*r)

    plt.figure(fig.number)
    plt.annotate(full_stamp_str, (0.99, 0.01),
                 xycoords='figure fraction',
                 fontsize='x-small',
                 ha='right',
                 va='bottom',
                 **kwargs)


def save_figures(filename=None, figs=None, dpi=200, fmt='pdf', directory=None,
                 add_filename_timestamp=True, stamp_kwargs=None):
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    if filename is None:
        filename = 'all_figures'

    if add_filename_timestamp:
        filename = filename + '_' + time.strftime('%y%m%d-%H%M')

    if figs is None:
        figs = [plt.figure(n) for n in plt.get_fignums()]
    else:
        try:
            _ = iter(figs)
        except TypeError:
            figs = [figs, ]

    if stamp_kwargs:
        [stamp_fig(fig, **stamp_kwargs) for fig in figs]

    directory = 'figures' if directory is None else directory
    csutils.touchdir(directory)

    file_path = os.path.join(directory, filename)
    if fmt == 'pdf':
        file_path += '.pdf'
        with PdfPages(file_path) as pp:
            for fig in figs:
                fig.savefig(pp, format='pdf', dpi=dpi)
    else:
        for i, fig in enumerate(figs):
            fig.savefig('%s_%d.%s' % (file_path, i, fmt), format=fmt, dpi=dpi)


def set_mpl_defaults():
    import matplotlib as mpl

    tex_preamble = r'''
    \usepackage{sansmathfonts}
    \usepackage{helvet}
    \renewcommand{\rmdefault}{\sfdefault}
    \usepackage{units}
    '''
    mpl.rc('text', usetex=True)
    mpl.rc('text.latex', preamble=tex_preamble)
