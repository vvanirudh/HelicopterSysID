import matplotlib.pyplot as plt
import numpy as np
import matplotlib

from hover_sys_id import DEFAULT_NUM_SAMPLES_PER_ITERATION

YLIM_MIN = 50.0
YLIM_MAX = 60.0

matplotlib.rcParams['ps.useafm'] = True
matplotlib.rcParams['pdf.use14corefonts'] = True
matplotlib.rcParams['text.usetex'] = True
plt.rcParams.update({'font.size': 15})

def plot_data(seeds):
    ag_costs = []
    pdl_costs = []
    ag_lqr_calls = []
    pdl_lqr_calls = []
    best_cost = []
    ag_costs_model = []
    ag_expert_cost_model = []
    pdl_costs_model = []

    for seed in seeds:
        ag_costs.append(np.load(f"data/ag_costs_{seed}.npy"))
        pdl_costs.append(np.load(f"data/pdl_costs_{seed}.npy"))
        ag_lqr_calls.append(np.load(f"data/ag_lqr_calls_{seed}.npy"))
        pdl_lqr_calls.append(np.load(f"data/pdl_lqr_calls_{seed}.npy"))
        best_cost.append(np.load(f"data/best_cost_{seed}.npy"))
        ag_costs_model.append(np.load(f"data/ag_costs_model_{seed}.npy"))
        ag_expert_cost_model.append(np.load(f"data/ag_expert_cost_model_{seed}.npy"))
        pdl_costs_model.append(np.load(f"data/pdl_costs_model_{seed}.npy"))

    ag_mean_costs, ag_std_costs = np.mean(ag_costs, axis=0), np.std(ag_costs, axis=0) / len(seeds)
    pdl_mean_costs, pdl_std_costs = np.mean(pdl_costs, axis=0), np.std(pdl_costs, axis=0) / len(seeds)
    ag_mean_lqr_calls, ag_std_lqr_calls = np.mean(ag_lqr_calls, axis=0), np.std(ag_lqr_calls, axis=0) / len(seeds)
    pdl_mean_lqr_calls , pdl_std_lqr_calls= np.mean(pdl_lqr_calls, axis=0), np.std(pdl_lqr_calls, axis=0) / len(seeds)
    best_mean_cost, best_std_cost = np.mean(best_cost), np.std(best_cost) / len(seeds)
    ag_mean_costs_model, ag_std_costs_model = np.mean(ag_costs_model, axis=0), np.std(ag_costs_model, axis=0) / len(seeds)
    pdl_mean_costs_model, pdl_std_costs_model = np.mean(pdl_costs_model, axis=0), np.std(pdl_costs_model, axis=0) / len(seeds)
    ag_mean_expert_cost_model, ag_std_expert_cost_model = np.mean(ag_expert_cost_model, axis=0), np.std(ag_expert_cost_model, axis=0) / len(seeds)

    best_mean_costs = np.array([best_mean_cost for _ in range(len(ag_mean_costs))])
    best_std_costs = np.array([best_std_cost for _ in range(len(ag_mean_costs))])

    plt.clf()
    xrange = np.arange(len(ag_mean_costs)) * DEFAULT_NUM_SAMPLES_PER_ITERATION
    plt.plot(xrange, ag_mean_costs, label="SysID", color="blue")
    plt.fill_between(xrange, ag_mean_costs - ag_std_costs, ag_mean_costs + ag_std_costs, color="blue", alpha=0.2)
    plt.plot(xrange, pdl_mean_costs, label="Efficient SysID", color="green")
    plt.fill_between(xrange, pdl_mean_costs - pdl_std_costs, pdl_mean_costs + pdl_std_costs, color="green", alpha=0.2)
    plt.plot(xrange, best_mean_costs, label="Opt", color="red")
    plt.fill_between(xrange, best_mean_costs - best_std_costs, best_mean_costs + best_std_costs, color="red", alpha=0.2)
    plt.ylim([YLIM_MIN, YLIM_MAX])
    plt.xlabel("Number of real world interactions")
    plt.ylabel("Cost of policy in real world")
    plt.legend()
    plt.grid(True)
    #plt.title("Cost of policy in real world vs number of real world interactions")
    plt.savefig("hover_exp.png")

    plt.clf()
    plt.plot(xrange, ag_mean_lqr_calls, label="SysID", color="blue")
    plt.fill_between(xrange, ag_mean_lqr_calls - ag_std_lqr_calls, ag_mean_lqr_calls + ag_std_lqr_calls, color="blue", alpha=0.2)
    plt.plot(xrange, pdl_mean_lqr_calls, label="Efficient SysID", color="green")
    plt.fill_between(xrange, pdl_mean_lqr_calls - pdl_std_lqr_calls, pdl_mean_lqr_calls + pdl_std_lqr_calls, color="green", alpha=0.2)
    plt.xlabel("Number of real world interactions")
    plt.ylabel("Number of LQR solver calls")
    plt.legend()
    plt.grid(True)
    #plt.title("Number of LQR solver calls vs number of real world interactions")
    plt.savefig("hover_exp_lqr_calls.png")

    plt.clf()
    plt.plot(xrange, ag_mean_costs_model, label="SysID", color="blue")
    plt.fill_between(xrange, ag_mean_costs_model - ag_std_costs_model, ag_mean_costs_model + ag_std_costs_model, color="blue", alpha=0.2)
    plt.plot(xrange, pdl_mean_costs_model, label="Efficient SysID", color="green")
    plt.fill_between(xrange, pdl_mean_costs_model - pdl_std_costs_model, pdl_mean_costs_model + pdl_std_costs_model, color="green", alpha=0.2)
    plt.plot(xrange, ag_mean_expert_cost_model, label="Expert", color="orange")
    plt.fill_between(xrange, ag_mean_expert_cost_model - ag_std_expert_cost_model, ag_mean_expert_cost_model + ag_std_expert_cost_model, color="orange", alpha=0.2)
    plt.ylim([YLIM_MIN, YLIM_MAX])
    plt.xlabel("Number of real world interactions")
    plt.ylabel("Cost of policy in model")
    plt.legend()
    plt.grid(True)
    #plt.title("Cost of policy in model vs number of real world interactions")
    plt.savefig("hover_exp_planning_error.png")
    



def plot_hover_exp_costs_lqr_calls(seed: int):
    ag_costs = np.load(
        f"data/ag_costs_{seed}.npy"
    )
    pdl_costs = np.load(
        f"data/pdl_costs_{seed}.npy")
    ag_lqr_calls = np.load(
        f"data/ag_lqr_calls_{seed}.npy"
    )
    pdl_lqr_calls = np.load(
        f"data/pdl_lqr_calls_{seed}.npy"
    )
    best_cost = np.load(
        f"data/best_cost_{seed}.npy"
    )[0]

    xrange = np.arange(len(ag_costs)) * DEFAULT_NUM_SAMPLES_PER_ITERATION
    plt.clf()
    plt.plot(xrange, ag_costs,
             label="SysID", color="blue", alpha=1.0)
    # plt.plot(xrange, ag_costs, color="blue", alpha=0.2)
    plt.plot(
        xrange, pdl_costs, label="Efficient SysID", color="green", alpha=1.0
    )
    # plt.plot(xrange, pdl_costs, color="green", alpha=0.2)
    plt.plot(
        xrange,
        [best_cost for _ in range(len(ag_costs))],
        "--",
        label="Opt",
        color="red",
    )
    plt.legend()
    plt.xlabel("Number of real world interactions")
    plt.ylabel("Cost of Policy")
    # plt.yscale("log")
    plt.grid(True)
    plt.ylim([YLIM_MIN, YLIM_MAX])
    plt.savefig(f"hover_exp_{seed}.png")

    plt.clf()
    plt.plot(xrange, ag_lqr_calls, label="SysID", color="blue")
    plt.plot(xrange, pdl_lqr_calls, label="Efficient SysID", color="green")
    plt.legend()
    plt.xlabel("Number of real world interactions")
    plt.ylabel("Number of LQR solver calls")
    plt.savefig(f"hover_exp_lqr_calls_seed_{seed}.png")


def plot_hover_exp_planning_model_error(seed: int):
    ag_costs_model = np.load(f"data/ag_costs_model_{seed}.npy")
    ag_expert_cost_model = np.load(f"data/ag_expert_cost_model_{seed}.npy")
    ag_model_errors = np.load(f"data/ag_model_errors_{seed}.npy")

    pdl_costs_model = np.load(f"data/pdl_costs_model_{seed}.npy")
    pdl_expert_cost_model = np.load(f"data/pdl_expert_cost_model_{seed}.npy")
    pdl_model_errors = np.load(f"data/pdl_model_errors_{seed}.npy")

    xrange = np.arange(len(ag_costs_model)) * DEFAULT_NUM_SAMPLES_PER_ITERATION
    plt.clf()

    plt.plot(xrange, ag_costs_model, label="SysID cost in model")
    plt.plot(xrange, ag_expert_cost_model, label="SysID expert cost in model")
    plt.plot(xrange, pdl_costs_model, label="Efficient SysID cost in model")
    plt.legend()
    plt.xlabel("Number of real world interactions")
    plt.ylabel("Cost of policy")
    plt.ylim([YLIM_MIN, YLIM_MAX])
    plt.savefig(f"hover_exp_planning_error_seed_{seed}.png")

    plt.clf()

    plt.plot(xrange, ag_model_errors, label="SysID")
    plt.plot(xrange, pdl_model_errors, label="Efficient SysID")
    plt.legend()
    plt.xlabel("Number of real world interactions")
    plt.ylabel("Modeling error")
    plt.ylim([0, 0.002])
    plt.savefig(f"hover_exp_modeling_error_seed_{seed}.png")


if __name__ == "__main__":
    seeds = range(10)
    # for seed in seeds:
    #     plot_hover_exp_costs_lqr_calls(seed)
    #     plot_hover_exp_planning_model_error(seed)

    plot_data(seeds)
