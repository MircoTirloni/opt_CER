# Import modules
import matplotlib.pyplot as plt
import numpy as np
import pyswarms as ps
from sim_CER_script import sim_CER, n_sim, n_car, n_dati, save_excel_path, excel_data_path_save_premi, \
    ex_data_path_template
from pyswarms.utils.plotters import (plot_cost_history)

"""
            Ottimizzazione con combinazione utenze fissata  
"""

"""
inserire codice che simula una combinazione di utenze con un p/c ratio in input
e da come valore in output il parametro da ottimizzare

Funzione da ottimizzare:

sim_CER(j, n_car, n_dati, excel_data_path_save_ris, excel_data_path_save_premi, ex_data_path_template,
            save_txt_file_path, num_comb,n_p_c_ratio)

k = pros_cons_ratio [variabile indipendente, range: (min=0.01, max=1.5)]

p/c=0.01 : 1%   di prosumer sul tot edifici
p/c=1.5  : 60%  di prosumer sul tot edifici

p_opt va calcolato dentro sim_CER   NB: a fine funzione aggiungere "return p_OPT"

"""


def ratio_eval(p_pos_RES, p_pos_OFF, p_pos_IND, p_pos_SHO):
    """       NB: ratio_RES + ratio_OFF + ratio_IND + ratio_SHO = 1   """

    ratio_RES = p_pos_RES
    ratio_OFF = (1 - p_pos_RES) * p_pos_OFF
    ratio_IND = (1 - p_pos_RES) * (1 - p_pos_OFF) * p_pos_IND
    ratio_SHO = 1 - (ratio_RES + ratio_OFF + ratio_IND)
    print("ratio utenze")
    print(ratio_RES, ratio_OFF, ratio_IND, ratio_SHO)

    return ratio_RES, ratio_OFF, ratio_IND, ratio_SHO


def sim_CER_to_opt(X):
    # Parametro da ottimizzare
    p_opt = []
    # X è un vettore che contiene le posizioni delle particelle: denom_p_c_ratio, ratio_RES, ratio_OFF, ratio_IND, ratio_SHO
    n_particles = X.shape[0]  # number of particles
    # print(X.shape[0])  # number of elements along the first dimension of X
    # print(X)

    for i in range(n_particles):
        "lettura posizioni particella"
        p_pos_RES = X[i, 1]
        p_pos_OFF = X[i, 2]
        p_pos_IND = X[i, 3]
        p_pos_SHO = X[i, 4]

        denom_p_c_ratio = round(X[i, 0])
        print("p/c ratio")
        print(denom_p_c_ratio)
        ratio_RES, ratio_OFF, ratio_IND, ratio_SHO = ratio_eval(p_pos_RES, p_pos_OFF, p_pos_IND, p_pos_SHO)

        sum_ratio = ratio_RES + ratio_OFF + ratio_IND + ratio_SHO

        if sum_ratio == 1:
            p_opt.append(
                sim_CER(n_sim, n_car, n_dati, save_excel_path, excel_data_path_save_premi, ex_data_path_template,
                        denom_p_c_ratio, ratio_RES, ratio_OFF, ratio_IND, ratio_SHO))
        else:
            p_opt.append(0)

        print("p_opt=", p_opt)
        print("positions=", X[i])
        print("----------------------------------------------------------------------------------------------")
    return np.array(p_opt)


# Create bounds
max_bound = np.array([20, 1, 1, 1, 1])  # Convert to array
min_bound = np.array([0.51, 0, 0, 0, 0])
bounds = (min_bound, max_bound)

# Set-up PSO hyperparameters
"""
c1: The cognitive component speaks of the particle’s bias towards its personal best from its past experience (i.e., how attracted it is to its own best position
c2: The social component controls how the particles are attracted to the best score found by the swarm (i.e., the global best)
w:  is the inertia weight that controls the “memory” of the swarm’s previous position
"""
options = {'c1': 0.5, 'c2': 0.3, 'w': 0.92}

# Define initial positions of particles
"initial_positions is a two-dimensional array with shape (number_of_particles, number_of_dimensions)"
initial_positions = np.array([[1, 0.25, 0.33, 0.5, 0.5], [2, 0.25, 0.33, 0.5, 0.5],
                              [3, 0.25, 0.33, 0.5, 0.5], [4, 0.25, 0.33, 0.5, 0.5],
                              [5, 0.25, 0.33, 0.5, 0.5], [6, 0.25, 0.33, 0.5, 0.5],
                              [7, 0.25, 0.33, 0.5, 0.5], [8, 0.25, 0.33, 0.5, 0.5],
                              [9, 0.25, 0.33, 0.5, 0.5], [10, 0.25, 0.33, 0.5, 0.5]])

# Optimizer "option handling" strategy
oh_strategy = {"w": 'exp_decay', 'c1': 'lin_variation', "c2": 'lin_variation'}

# Call instance of PSO with initial positions
optimizer = ps.single.GlobalBestPSO(n_particles=10, dimensions=5, options=options, oh_strategy=oh_strategy,
                                    bounds=bounds,
                                    init_pos=initial_positions)

# optimizer = ps.single.GlobalBestPSO(n_particles=100, dimensions=5, options=options, bounds=bounds)
# kwargs = {"denom_p_c_ratio": 0, "ratio_RES": 0, "ratio_OFF": 0, "ratio_IND": 0, "ratio_SHO": 0}

# Perform optimization

cost, pos = optimizer.optimize(sim_CER_to_opt, iters=70)

# plot cost historykll +

fig, ax_h = plt.subplots(ncols=1, nrows=1)
fig.set_size_inches(15, 7)
plot_cost_history(cost_history=optimizer.cost_history, ax=ax_h)
ax_h.set_title("Cost history with exponential intertia decay")
plt.show()

"       Tempo previsto con 10 particelle: 15 min/iterazione    "
"                             60 iters = 15h                "

"""

pyswarms.single.global_best: 100%|██████████|70/70, best_cost=0.99
2024-06-06 21:46:27,184 - pyswarms.single.global_best - INFO
 - Optimization finished | best cost: 0.9900896834023252,
  best pos: [1.35176445 0.85722521 0.99490973 0.61190728 0.73831514]

pyswarms.single.global_best: 100%|██████████|70/70, best_cost=0.824
2024-06-08 10:51:33,054 - pyswarms.single.global_best - INFO 
- Optimization finished | best cost: 0.8243013832846513
, best pos: [1.06024631 0.80973461 0.99803962 0.41907258 0.83075922]

"""
