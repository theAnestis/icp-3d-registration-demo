from matplotlib import pyplot as plt
import orjson


def plot_time_comparison():
    """
    Plots a comparison of time spent per iteration.
    """
    optim_times = orjson.loads(open("./res_optim.json", "rb"))["time"]
    simple_times = orjson.loads(open("./res_simple.json", "rb"))["time"]

    fig, ax = plt.subplots(figsize=(6, 10))

    ax.plot(range(len(optim_times)), optim_times, "r", label="Optimised ICP", linewidth=2)
    ax.plot(range(len(simple_times)), simple_times, "b", label="Simple ICP", linewidth=2)
    ax.set_yscale("log")
    ax.set_xlabel("Iteration", fontsize=12)
    ax.set_ylabel("Time (ms)", fontsize=12)
    ax.set_title("Performance Comparison of ICP Variations", fontsize=14)
    ax.legend()

    plt.grid(True, which="both", linestyle="--", linewidth=0.5)

    plt.show()


def plot_rmse_comparison():
    """
    Plots a comparison of rmse per iteration.
    """
    optim_rmse = orjson.loads(open("./res_optim.json", "rb"))["rmse"]
    simple_rmse = orjson.loads(open("./res_simple.json", "rb"))["rmse"]

    fig, ax = plt.subplots(figsize=(6, 10))

    ax.plot(range(len(optim_rmse)), optim_rmse, "r", label="Optimised ICP", linewidth=2)
    ax.plot(range(len(simple_rmse)), simple_rmse, "b", label="Simple ICP", linewidth=2)

    ax.set_yscale("log")

    ax.set_xlabel("Iteration", fontsize=12)
    ax.set_ylabel("RMSE (Log Scale)", fontsize=12)
    ax.set_title("Convergence Comparison of ICP Variations", fontsize=14)
    ax.legend()

    plt.grid(True, which="both", linestyle="--", linewidth=0.5)

    plt.show()


plot_time_comparison()
plot_rmse_comparison()
