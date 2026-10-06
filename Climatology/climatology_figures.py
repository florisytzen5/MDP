"""
Report figures for the Sisal baseline climatology.

Data: ERA5-Land daily, cell c1 (centred 21.1 N, 90.0 W), and CHIRPS daily for
the same cell. Reference period 1991-2020; trends from 1979 onward.

Usage (in a notebook, with this file in the same folder):

    import climatology_figures as cf
    wx     = cf.load_era5land('Sisal_ERA5Land_daily_all.csv')
    chirps = cf.load_chirps('CHIRPS_daily_c1.csv')
    figs   = cf.make_all(wx, chirps, outdir='figures')

Each figure is also available on its own (fig_climate_diagram, fig_daily_temperature,
fig_annual_anomalies, fig_era5_vs_chirps) and returns a matplotlib Figure.
Re-running make_all on an updated CSV regenerates every figure, which is what
the annual update needs.
"""

import os

import matplotlib as mpl
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

# --------------------------------------------------------------------------
# Settings
# --------------------------------------------------------------------------
REF = (1991, 2020)          # climatological reference period (WMO normal)
TREND_START = 1979          # pre-1979 reanalysis is less constrained by observations
HALF_WINDOW = 15            # +/- days pooled for the daily climatology
RUNNING_MEAN = 11           # years, for the anomaly plot
FIG_WIDTH = 6.5             # inches, fits a single-column A4 report

# Colours follow the job they do; the same entity keeps the same colour in
# every figure. The two categorical pairs were checked for colour-blind
# separation and contrast on white.
COLORS = {
    'tmin':   '#2a78d6',    # blue
    'tmean':  '#3d3c39',    # near-black ink
    'tmax':   '#e34948',    # red
    'era5l':  '#4a3aa7',    # violet
    'chirps': '#eb6834',    # orange
    'warm':   '#e34948',    # anomaly > 0
    'cool':   '#2a78d6',    # anomaly < 0
    'ink':    '#0b0b0b',
    'ink2':   '#52514e',
    'grid':   '#e4e3df',
    'shade':  '#efeeea',
}

TEMP_VARS = [('t2m_min_c', 'Tmin', 'tmin'),
             ('t2m_c',     'Tmean', 'tmean'),
             ('t2m_max_c', 'Tmax', 'tmax')]

MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
          'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']


def set_style():
    """Consistent, quiet styling for all report figures."""
    mpl.rcParams.update({
        'font.size': 9,
        'axes.titlesize': 10,
        'axes.labelsize': 9,
        'axes.labelcolor': COLORS['ink2'],
        'axes.edgecolor': COLORS['ink2'],
        'axes.linewidth': 0.8,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.grid': True,
        'axes.axisbelow': True,
        'grid.color': COLORS['grid'],
        'grid.linewidth': 0.6,
        'xtick.color': COLORS['ink2'],
        'ytick.color': COLORS['ink2'],
        'legend.frameon': False,
        'legend.fontsize': 8.5,
        'lines.linewidth': 1.8,
        'figure.dpi': 120,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
    })


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------
def load_era5land(path, cell='c1'):
    """Daily ERA5-Land for one cell, with the cleaning decided in the audit:
    only temperature and precipitation kept, tiny negative precipitation set to 0."""
    df = pd.read_csv(path, parse_dates=['date']).set_index('date').sort_index()
    cols = ['t2m_c', 't2m_min_c', 't2m_max_c', 'precip_mm']
    wx = (df[[f'{c}_{cell}' for c in cols]]
          .rename(columns=lambda s: s.removesuffix(f'_{cell}')))
    wx['precip_mm'] = wx['precip_mm'].clip(lower=0)
    return wx


def load_chirps(path):
    """Daily CHIRPS precipitation (mm) for the same cell."""
    return (pd.read_csv(path, parse_dates=['date'])
              .set_index('date').sort_index()['precip_chirps_mm'])


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def _in_period(obj, period):
    return obj.loc[f'{period[0]}-01-01':f'{period[1]}-12-31']


def _doy365(index):
    """Day of year on a 365-day calendar (leap-year Mar-Dec shifted back by one)."""
    shift = (index.is_leap_year & (index.month > 2)).astype(int)
    return np.asarray(index.dayofyear - shift)


def doy_climatology(series, half=HALF_WINDOW, q=(10, 90)):
    """Mean and percentiles per day of year, pooled over a circular +/-half-day window.
    Feb 29 is dropped."""
    s = series.dropna()
    s = s[~((s.index.month == 2) & (s.index.day == 29))]
    doy, vals = _doy365(s.index), s.to_numpy()
    rows = []
    for d in range(1, 366):
        dist = np.abs(doy - d)
        dist = np.minimum(dist, 365 - dist)
        v = vals[dist <= half]
        rows.append([v.mean(), *np.percentile(v, q)])
    return pd.DataFrame(rows, index=pd.RangeIndex(1, 366, name='doy'),
                        columns=['mean', f'p{q[0]}', f'p{q[1]}'])


def monthly_precip_stats(daily, period=REF):
    """Median and 10th/90th percentile of monthly totals, per calendar month."""
    tot = _in_period(daily, period).resample('MS').sum()
    g = tot.groupby(tot.index.month)
    return pd.DataFrame({'median': g.median(),
                         'p10': g.quantile(0.10),
                         'p90': g.quantile(0.90)})


def sen_trend(y):
    """Sen's slope (per year) with 95% CI, intercept, and Mann-Kendall p-value.
    y: Series indexed by year."""
    y = y.dropna()
    x = y.index.to_numpy(dtype=float)
    slope, intercept, lo, hi = stats.theilslopes(y.to_numpy(), x, 0.95)
    p = stats.kendalltau(x, y.to_numpy()).pvalue
    return {'slope': slope, 'intercept': intercept, 'lo': lo, 'hi': hi, 'p': p}


def annual_temperature_anomalies(wx, ref=REF):
    ann = wx[[v for v, _, _ in TEMP_VARS]].resample('YS').mean()
    ann.index = ann.index.year
    return ann - ann.loc[ref[0]:ref[1]].mean()


# --------------------------------------------------------------------------
# Figure 1: climate diagram (monthly temperature + monthly precipitation)
# --------------------------------------------------------------------------
def fig_climate_diagram(wx, chirps, ref=REF, show_title=False):
    """Two panels sharing the month axis: monthly mean Tmin/Tmean/Tmax (top) and
    median monthly precipitation for ERA5-Land and CHIRPS with 10-90th percentile
    whiskers (bottom). Separate panels instead of a dual y-axis."""
    ref_wx = _in_period(wx, ref)
    mon = ref_wx.resample('MS').mean()
    tclim = mon.groupby(mon.index.month).mean()

    p_e = monthly_precip_stats(wx['precip_mm'], ref)
    p_c = monthly_precip_stats(chirps, ref)

    fig, (ax_t, ax_p) = plt.subplots(2, 1, figsize=(FIG_WIDTH, 5.6), sharex=True,
                                     gridspec_kw={'height_ratios': [1, 1], 'hspace': 0.12})
    x = np.arange(12)

    for var, label, key in reversed(TEMP_VARS):          # Tmax first in the legend
        ax_t.plot(x, tclim[var].to_numpy(), color=COLORS[key], marker='o', ms=4.5,
                  label=label)
        ax_t.annotate(f'{label}', xy=(x[-1], tclim[var].iloc[-1]), xytext=(8, 0),
                      textcoords='offset points', va='center', fontsize=8.5,
                      color=COLORS['ink2'])
    ax_t.set_ylabel('Air temperature at 2 m (°C)')
    ax_t.legend(loc='lower left', bbox_to_anchor=(0, 1.0), ncol=3)
    ax_t.margins(y=0.08)

    w = 0.38
    for off, stats_, key, label in [(-w / 2, p_e, 'era5l', 'ERA5-Land'),
                                    (w / 2, p_c, 'chirps', 'CHIRPS')]:
        med = stats_['median'].to_numpy()
        yerr = [med - stats_['p10'].to_numpy(), stats_['p90'].to_numpy() - med]
        ax_p.bar(x + off, med, width=w, color=COLORS[key], edgecolor='white',
                 linewidth=1.0, label=f'{label} (median, 10–90th pct.)')
        ax_p.errorbar(x + off, med, yerr=yerr, fmt='none', ecolor=COLORS['ink2'],
                      elinewidth=0.8, capsize=2)
    ax_p.set_ylabel('Precipitation (mm per month)')
    ax_p.set_xticks(x, MONTHS)
    ax_p.grid(axis='x', visible=False)
    ax_p.legend(loc='upper left')

    if show_title:
        ax_t.set_title(f'Monthly climatology, ERA5-Land cell c1 ({ref[0]}–{ref[1]})')
    return fig


# --------------------------------------------------------------------------
# Figure 2: daily temperature climatology
# --------------------------------------------------------------------------
def fig_daily_temperature(wx, ref=REF, half=HALF_WINDOW, show_title=False):
    """Day-of-year mean and 10-90th percentile band of Tmin, Tmean and Tmax."""
    ref_wx = _in_period(wx, ref)
    x = pd.Timestamp('2001-01-01') + pd.to_timedelta(np.arange(365), unit='D')

    fig, ax = plt.subplots(figsize=(FIG_WIDTH, 3.4))
    for var, label, key in reversed(TEMP_VARS):
        c = doy_climatology(ref_wx[var], half=half)
        ax.fill_between(x, c['p10'], c['p90'], color=COLORS[key], alpha=0.16, lw=0)
        ax.plot(x, c['mean'], color=COLORS[key], label=label)
    ax.set_ylabel('Air temperature at 2 m (°C)')
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
    ax.set_xlim(x[0], x[-1])
    ax.legend(loc='lower center', ncol=3, title=f'mean and 10–90th percentile (±{half}-day window)',
              title_fontsize=8)
    ax.margins(y=0.15)
    if show_title:
        ax.set_title(f'Daily temperature climatology, ERA5-Land cell c1 ({ref[0]}–{ref[1]})')
    return fig


# --------------------------------------------------------------------------
# Figure 3: annual temperature anomalies with trend
# --------------------------------------------------------------------------
def fig_annual_anomalies(wx, ref=REF, trend_start=TREND_START, var='t2m_c',
                         show_title=False):
    """Annual anomaly bars (warm/cool), 11-year running mean, Sen's slope trend
    from trend_start, and shading of the less reliable pre-trend_start period."""
    a = annual_temperature_anomalies(wx, ref)[var]
    last = int(a.index.max())
    tr = sen_trend(a.loc[trend_start:])

    fig, ax = plt.subplots(figsize=(FIG_WIDTH, 3.3))
    ax.axvspan(a.index.min() - 0.5, trend_start - 0.5, color=COLORS['shade'], lw=0)
    shade_proxy = mpl.patches.Patch(facecolor=COLORS['shade'], edgecolor=COLORS['ink2'],
                                    linewidth=0.5,
                                    label=f'before {trend_start}: fewer observations, use with caution')
    ax.bar(a.index, a.to_numpy(),
           color=np.where(a > 0, COLORS['warm'], COLORS['cool']),
           width=0.82, edgecolor='white', linewidth=0.3)
    ax.plot(a.index, a.rolling(RUNNING_MEAN, center=True).mean(), color=COLORS['ink'],
            lw=1.8, label=f'{RUNNING_MEAN}-year running mean')
    xt = np.array([trend_start, last], dtype=float)
    ax.plot(xt, tr['intercept'] + tr['slope'] * xt, color=COLORS['ink2'], lw=1.4, ls='--',
            label=(f"trend {trend_start}–{last}: {tr['slope']*10:+.2f} °C/decade "
                   f"(95% CI {tr['lo']*10:+.2f} to {tr['hi']*10:+.2f})"))
    ax.axhline(0, color=COLORS['ink2'], lw=0.8)
    ax.set_ylabel(f'Anomaly vs {ref[0]}–{ref[1]} (°C)')
    ax.set_xlim(a.index.min() - 1, last + 1)
    ax.grid(axis='x', visible=False)
    handles, _ = ax.get_legend_handles_labels()
    ax.legend(handles=handles + [shade_proxy], loc='upper left',
              bbox_to_anchor=(0, -0.1), ncol=1, fontsize=8)
    if show_title:
        label = dict((v, l) for v, l, _ in TEMP_VARS)[var]
        ax.set_title(f'Annual {label} anomaly, ERA5-Land cell c1')
    return fig


# --------------------------------------------------------------------------
# Figure 4: ERA5-Land vs CHIRPS
# --------------------------------------------------------------------------
def fig_era5_vs_chirps(wx, chirps, period=REF, show_title=False):
    """Annual precipitation totals of both products (top) and their ratio (bottom).
    Also prints the agreement statistics used in the text."""
    both = pd.concat({'era5l': wx['precip_mm'], 'chirps': chirps}, axis=1)
    both = _in_period(both, period).dropna()
    ann = both.resample('YS').sum()
    ann.index = ann.index.year
    ratio = ann['era5l'] / ann['chirps']

    mon = both.resample('MS').sum()
    anom = mon - mon.groupby(mon.index.month).transform('mean')
    print(f"ERA5-Land vs CHIRPS {period[0]}–{period[1]}: "
          f"r monthly = {mon.corr().iloc[0, 1]:.2f}, "
          f"r monthly anomalies = {anom.corr().iloc[0, 1]:.2f}, "
          f"r annual = {ann.corr().iloc[0, 1]:.2f}, mean ratio = {ratio.mean():.2f}")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(FIG_WIDTH, 4.8), sharex=True,
                                   gridspec_kw={'height_ratios': [1.6, 1], 'hspace': 0.12})
    for key, label in [('era5l', 'ERA5-Land'), ('chirps', 'CHIRPS')]:
        ax1.plot(ann.index, ann[key], color=COLORS[key], marker='o', ms=4, label=label)
    ax1.set_ylabel('Annual precipitation (mm)')
    ax1.legend(loc='lower left', bbox_to_anchor=(0, 1.0), ncol=2)

    ax2.plot(ratio.index, ratio, color=COLORS['ink'], marker='o', ms=4)
    ax2.hlines(ratio.mean(), ratio.index.min(), ratio.index.max(),
               color=COLORS['ink2'], lw=1.2, ls='--')
    ax2.annotate(f'mean {ratio.mean():.2f}', xy=(ratio.index.max(), ratio.mean()),
                 xytext=(8, 0), textcoords='offset points', va='center',
                 fontsize=8.5, color=COLORS['ink2'])
    ax2.set_ylabel('ERA5-Land / CHIRPS')
    ax2.set_xlim(ratio.index.min() - 1, ratio.index.max() + 4)
    if show_title:
        ax1.set_title(f'Annual precipitation, ERA5-Land vs CHIRPS, cell c1 '
                      f'({period[0]}–{period[1]})')
    return fig


# --------------------------------------------------------------------------
# Everything at once
# --------------------------------------------------------------------------
def save(fig, name, outdir='figures'):
    """Save as PNG (Word/Docs) and PDF (vector, for LaTeX/Overleaf)."""
    os.makedirs(outdir, exist_ok=True)
    for ext in ('png', 'pdf'):
        fig.savefig(os.path.join(outdir, f'{name}.{ext}'))


def make_all(wx, chirps, outdir='figures', ref=REF, show_title=False):
    set_style()
    figs = {
        'fig1_climate_diagram':    fig_climate_diagram(wx, chirps, ref, show_title),
        'fig2_daily_temperature':  fig_daily_temperature(wx, ref, show_title=show_title),
        'fig3_annual_anomalies':   fig_annual_anomalies(wx, ref, show_title=show_title),
        'fig4_era5_vs_chirps':     fig_era5_vs_chirps(wx, chirps, ref, show_title),
    }
    for name, fig in figs.items():
        save(fig, name, outdir)
    print(f'saved {len(figs)} figures to {os.path.abspath(outdir)}')
    return figs
