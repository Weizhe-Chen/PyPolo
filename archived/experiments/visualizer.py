from pathlib import Path

from matplotlib import pyplot as plt
from matplotlib import ticker as ticker
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
from scipy import interpolate

plt.rcParams["image.origin"] = "lower"
plt.rcParams["image.cmap"] = "jet"
plt.rcParams["image.interpolation"] = "gaussian"
plt.rcParams["mathtext.fontset"] = "cm"
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.size"] = 10
plt.rcParams["axes.labelsize"] = 10
plt.rcParams["legend.fontsize"] = 8
plt.rcParams["xtick.labelsize"] = 8
plt.rcParams["ytick.labelsize"] = 8


def set_size(width=516, fraction=1.0):
    """Set figure dimensions to avoid scaling in LaTeX.

    Parameters
    ----------
    width: float
            Document textwidth or columnwidth in pts
    fraction: float, optional
            Fraction of the width which you wish the figure to occupy

    Returns
    -------
    fig_dim: tuple
            Dimensions of figure in inches
    """
    # Width of figure (in pts)
    fig_width_pt = width * fraction
    # Convert from pt to inches
    inches_per_pt = 1 / 72.27
    # Golden ratio to set aesthetic figure height
    # https://disq.us/p/2940ij3
    golden_ratio = (5**.5 - 1) / 2
    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt
    # Figure height in inches
    fig_height_in = fig_width_in * golden_ratio
    fig_dim = (fig_width_in, fig_height_in)
    return fig_dim


class OOMFormatter(ticker.ScalarFormatter):

    def __init__(self, order=0, fformat="%1.1f", offset=True, mathText=True):
        super().__init__(
            useOffset=offset,
            useMathText=mathText,
        )
        self.oom = order
        self.fformat = fformat

    def _set_order_of_magnitude(self):
        self.orderOfMagnitude = self.oom


def order_of_magnitude(array):
    return np.floor(np.log10(array.mean()))


def plot_line(ax, x, y):
    ax.plot(x, y)
    ax.grid("on")
    ax.autoscale(tight=True)


def plot_metrics(args, kernels, seeds, postfix=None, fraction=0.2):
    fig_dir = args.figure_dir + "/".join([args.env_name, args.strategy])
    Path(fig_dir).mkdir(parents=True, exist_ok=True)
    metrics = {1: "SMSE", 2: "MSLL", 3: "NLPD", 4: "RMSE", 5: "MAE"}
    xs = np.arange(args.num_init_samples, args.max_num_samples)
    for index, metric_name in metrics.items():
        fig, ax = plt.subplots(figsize=set_size(fraction=fraction))
        for kernel in kernels:
            multiple_runs = []
            for seed in seeds:
                experiment_id = "/".join([
                    str(seed),
                    args.env_name,
                    args.strategy,
                    kernel,
                ])
                save_dir = args.output_dir + experiment_id
                metrics = np.genfromtxt(f"{save_dir}/metrics.csv",
                                        delimiter=',',
                                        skip_header=1)
                num_samples = metrics[:, 0]
                interpolated = interpolate.interp1d(num_samples,
                                                    metrics[:, index])
                multiple_runs.append(interpolated(xs))
            multiple_runs = np.vstack(multiple_runs)
            mean = multiple_runs.mean(axis=0)
            std = multiple_runs.std(axis=0)
            ax.plot(xs, mean, alpha=0.8)
            ax.fill_between(
                xs,
                mean - 1.0 * std,
                mean + 1.0 * std,
                alpha=0.2,
            )
        ax.grid("on")
        ax.autoscale(tight=True)
        ax.yaxis.set_major_formatter(
            ticker.FuncFormatter(lambda x, _: "% .1f" % x))
        fig.tight_layout()
        name = metric_name if postfix is None else metric_name + postfix
        fig.savefig(
            f"{fig_dir}/{name}.pdf",
            format='pdf',
            bbox_inches='tight',
        )


def create_colorbar_ax(ax):
    cax = make_axes_locatable(ax).append_axes(
        "right",
        size="5%",
        pad=0.05,
    )
    return cax


def get_matplotlib_axes():
    fig, axes = plt.subplots(2, 2, figsize=(9, 8), sharex=True, sharey=True)
    fig.subplots_adjust(
        top=0.9,
        bottom=0.1,
        left=0.1,
        right=0.9,
        hspace=0.2,
        wspace=0.2,
    )
    caxes = []
    for ax in axes.ravel():
        caxes.append(create_colorbar_ax(ax))
    caxes = np.asarray(caxes).reshape(2, 2)
    return axes, caxes


def clear_axes(axes, caxes):
    for each in axes.ravel()[1:]:
        each.cla()
    for each in caxes.ravel()[1:]:
        each.cla()


def set_limits(axes, args):
    for ax in axes.ravel()[1:]:
        ax.set_xlim(args.env_extent[:2])
        ax.set_ylim(args.env_extent[2:])


def plot_image(args, ax, cax, values, title):
    matrix = values.reshape(args.eval_grid)
    im = ax.imshow(
        matrix,
        extent=args.env_extent
        if title == "Ground Truth" else args.task_extent,
    )
    plt.colorbar(
        im,
        cax=cax,
        format=OOMFormatter(order_of_magnitude(values)),
    )
    workspace = plt.Rectangle(
        (args.task_extent[0], args.task_extent[2]),
        args.task_extent[1] - args.task_extent[0],
        args.task_extent[3] - args.task_extent[2],
        linewidth=3,
        edgecolor="white",
        alpha=0.8,
        fill=False,
    )
    ax.add_patch(workspace)
    ax.set_title(title)


def pause():
    plt.gcf().canvas.mpl_connect(
        "key_release_event",
        lambda event: [exit(0) if event.key == "escape" else None],
    )
    plt.pause(1e-2)


def plot_line_with_uncertainty(ax, x, mean, std, color):
    ax.plot(x, mean, color=color)
    ax.fill_between(
        x,
        mean - std,
        mean + std,
        color=color,
        alpha=0.3,
    )


def plot_benchmarking_metrics(
    args,
    seeds,
    order_of_magnitudes,
):
    env_name = args.env.split('.')[0]
    figure_dir = "/".join([args.figure_dir, env_name])
    kernel_colors = {"rbf": 'k', "ak": 'r', "gibbs": 'g', "dkl": 'b'}
    num_data = None
    fig, ax = plt.subplots(1, 3, sharex=True)
    for kernel in kernel_colors.keys():
        nlpds, rmses, maes = [], [], []
        for seed in seeds:
            save_dir = "/".join([
                args.output_dir,
                env_name,
                kernel,
                str(seed),
            ])
            metrics = np.genfromtxt(
                f"{save_dir}/metrics.csv",
                delimiter=',',
                skip_header=1,
            )
            if num_data is None:
                num_data = metrics[:, 0]
            nlpds.append(metrics[:, 1])
            rmses.append(metrics[:, 2])
            maes.append(metrics[:, 3])
        nlpds = np.vstack(nlpds)
        rmses = np.vstack(rmses)
        maes = np.vstack(maes)
        plot_line_with_uncertainty(
            ax[0],
            num_data,
            nlpds.mean(axis=0),
            nlpds.std(axis=0),
            kernel_colors[kernel],
        )
        plot_line_with_uncertainty(
            ax[1],
            num_data,
            rmses.mean(axis=0),
            rmses.std(axis=0),
            kernel_colors[kernel],
        )
        plot_line_with_uncertainty(
            ax[2],
            num_data,
            maes.mean(axis=0),
            maes.std(axis=0),
            kernel_colors[kernel],
        )
    ylabels = ["NLPD", "RMSE", "MAE"]
    for index, each in enumerate(ax):
        each.set_title(ylabels[index], loc="right")
        each.yaxis.set_major_formatter(OOMFormatter(
            order_of_magnitudes[index]))
        each.grid("on")
        each.autoscale(tight=True)
    fig.tight_layout()
    Path(figure_dir).mkdir(parents=True, exist_ok=True)
    #  fig.savefig(f"{args.figure_dir}metrics.pdf")
    fig.savefig(f"{figure_dir}/metrics.png", dpi=300)


def plot_benchmarking_metrics_interpolated(
    env_name,
    strategy,
    kernel_colors,
    seeds,
    num_samples,
    order_of_magnitudes,
):
    fig, ax = plt.subplots(1, 3, sharex=True)
    for kernel in kernel_colors.keys():
        nlpds, rmses, maes = [], [], []
        for seed in seeds:
            metrics_dir = "/".join([
                "outputs",
                str(seed),
                f"{env_name}",
                f"{strategy}",
                kernel,
            ])
            metrics = np.genfromtxt(
                f"{metrics_dir}/metrics.csv",
                delimiter=',',
                skip_header=1,
            )
            num_data = metrics[:, 0]
            nlpd = metrics[:, 1]
            nlpd_fn = interpolate.interp1d(num_data, nlpd)
            rmse = metrics[:, 2]
            rmse_fn = interpolate.interp1d(num_data, rmse)
            mae = metrics[:, 3]
            mae_fn = interpolate.interp1d(num_data, mae)
            nlpds.append(nlpd_fn(num_samples))
            rmses.append(rmse_fn(num_samples))
            maes.append(mae_fn(num_samples))
        nlpds = np.vstack(nlpds)
        rmses = np.vstack(rmses)
        maes = np.vstack(maes)
        plot_line_with_uncertainty(
            ax[0],
            num_samples,
            nlpds.mean(axis=0),
            nlpds.std(axis=0),
            kernel_colors[kernel],
        )
        plot_line_with_uncertainty(
            ax[1],
            num_samples,
            rmses.mean(axis=0),
            rmses.std(axis=0),
            kernel_colors[kernel],
        )
        plot_line_with_uncertainty(
            ax[2],
            num_samples,
            maes.mean(axis=0),
            maes.std(axis=0),
            kernel_colors[kernel],
        )
    ylabels = ["NLPD", "RMSE", "MAE"]
    for index, each in enumerate(ax):
        each.set_title(ylabels[index], loc="right")
        each.yaxis.set_major_formatter(OOMFormatter(
            order_of_magnitudes[index]))
        each.grid("on")
        each.autoscale(tight=True)
    fig.tight_layout()
    Path("./results/").mkdir(parents=True, exist_ok=True)
    fig.savefig(f"./results/{env_name}_{strategy}.png", dpi=300)


def plot_learning_curve(
    metrics_train,
    name_train,
    metrics_test,
    name_test,
    fraction,
):
    fig, ax1 = plt.subplots(figsize=set_size(fraction=fraction))
    ax2 = ax1.twinx()
    color_train, color_test = "tab:blue", "tab:red"
    ax1.plot(range(len(metrics_train)), metrics_train, color=color_train)
    ax1.set_ylabel(name_train, color=color_train)
    ax1.tick_params(axis='y', labelcolor=color_train)
    ax2.plot(range(len(metrics_test)), metrics_test, color=color_test)
    ax2.set_ylabel(name_test, color=color_test)
    ax2.tick_params(axis='y', labelcolor=color_test)
    ax1.grid("on")
    ax1.autoscale(tight=True)
    return fig, ax1
