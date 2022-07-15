import numpy as np
import copy
from train_eval import *


def gen_conn(in_dim, out_dim):
    # connection is a 2 x n matrix, the 1st row is the row index of the weight matrix,
    # the 2nd row is the idx of column index of weight matrix
    row_dix = np.repeat(np.arange(out_dim), in_dim).reshape(1, -1)
    col_idx = np.tile(np.arange(in_dim), out_dim).reshape(1, -1)
    connections = np.concatenate((row_dix, col_idx))
    idx_list = range(connections.shape[1])
    return connections, idx_list


def gen_indices(i, idx_list, s_floor, s_ceil, pop_size, length):
    return np.unique(
        random.sample(idx_list, round((s_floor + i * (s_ceil - s_floor) / (pop_size - 1)) * length))).astype(int)


def gen_init_pop(init_sizes, idx_list, s_range, lyr_num):
    indi = dict()
    pop = dict()

    pop_key = 0
    if lyr_num == 3:
        for i in range(init_sizes[0]):
            indi[0] = gen_indices(i, idx_list[0], s_range[0, 0], s_range[1, 0], init_sizes[0], len(idx_list[0]))
            for j in range(init_sizes[1]):
                indi[1] = gen_indices(j, idx_list[1], s_range[0, 1], s_range[1, 1], init_sizes[1], len(idx_list[1]))
                for k in range(init_sizes[2]):
                    indi[2] = gen_indices(k, idx_list[2], s_range[0, 2], s_range[1, 2], init_sizes[2], len(idx_list[2]))
                    pop[pop_key] = copy.deepcopy(indi)
                    pop_key += 1
    else:
        for i in range(init_sizes[0]):
            indi[0] = gen_indices(i, idx_list[0], s_range[0, 0], s_range[1, 0], init_sizes[0], len(idx_list[0]))
            for j in range(init_sizes[1]):
                indi[1] = gen_indices(j, idx_list[1], s_range[0, 1], s_range[1, 1], init_sizes[1], len(idx_list[1]))
                for k in range(init_sizes[2]):
                    indi[2] = gen_indices(k, idx_list[2], s_range[0, 2], s_range[1, 2], init_sizes[2], len(idx_list[2]))
                    for m in range(init_sizes[3]):
                        indi[3] = gen_indices(m, idx_list[3], s_range[0, 3], s_range[1, 3], init_sizes[3],
                                              len(idx_list[3]))
                        pop[pop_key] = copy.deepcopy(indi)
                        pop_key += 1
    return pop


def indi2mask(indi, feat_num, lyr_1_out, lyr_2_out, lyr_3_out):
    # generate mask
    mask = dict()
    i = 0
    for in_dim, out_dim in [(feat_num, lyr_1_out),
                            (lyr_1_out, lyr_2_out),
                            (lyr_2_out, lyr_3_out),
                            (lyr_3_out, 2)]:
        mask[i] = np.zeros((out_dim * in_dim,))
        mask[i][indi[i]] = 1
        mask[i] = mask[i].reshape(out_dim, in_dim)
        i += 1
        if lyr_3_out == 2 and i == 3:
            break
    return mask


def mask2indi(mask):
    # generate individual
    indi = dict()
    for i in range(len(mask.keys())):
        mask[i] = mask[i].flatten()
        indi[i] = np.where(mask[i] == 1)[0]
    return indi


def cal_indi_fitness(indi, feat_num, lyr_1_out, lyr_2_out, lyr_3_out, dnn_param, save_dnn):
    c = 0.05
    mask = indi2mask(indi, feat_num, lyr_1_out, lyr_2_out, lyr_3_out)
    avg_vacc, avg_tacc, avg_trloss, avg_vloss = train_and_eval('sc', mask, dnn_param[0],
                                                               dnn_param[1], dnn_param[2], dnn_param[3], save_dnn)
    fitness = avg_vacc - c * avg_trloss
    fitness_arr = np.array([[fitness], [avg_tacc]])
    return fitness_arr


def cal_pop_fitness(pop, feat_num, lyr_1_out, lyr_2_out, lyr_3_out, dnn_param):
    pop_size = len(pop.keys())
    fit_arr = np.zeros((2, pop_size))
    for indi in range(pop_size):
        fit_arr[:, indi] = cal_indi_fitness(pop[indi], feat_num, lyr_1_out, lyr_2_out, lyr_3_out, dnn_param,
                                            save_dnn=0).reshape(-1, )
    return fit_arr


def psm(init_pop_size, init_fit_arr, fit_arr, pop_size_cnt, beta):
    '''
    Population Shrinking mechanism (PSM)
    Shrinking population size according to the standard deviation of the population fitness
    '''
    init_std = np.std(init_fit_arr[0, :])
    init_std = max(init_std, 0.01)  # too small init_std would throw exceptions
    new_std = np.std(fit_arr[0, :])

    alpha = np.exp((np.log(init_pop_size) / init_std))

    x_c = (init_std) / 2
    y_c = (1 + init_pop_size) / 2
    new_pop_size = np.floor(- alpha ** (2 * x_c - new_std) + 2 * y_c)

    if new_pop_size >= pop_size_cnt[0]:
        pop_size_cnt[1] += 1
    else:
        pop_size_cnt[0] = new_pop_size
        pop_size_cnt[1] = 0

    print('init_std: ', init_std, '\tnew_std: ', new_std)
    print('pop_size_cnt:', pop_size_cnt)

    return int(pop_size_cnt[0])


def roulette(fit_arr):
    '''roulette selection'''
    prob_arr = fit_arr - np.min(fit_arr) + 0.1 * np.max(fit_arr - np.min(fit_arr))
    random_val = random.uniform(0, sum(prob_arr))
    probability = 0
    for i in range(len(fit_arr)):
        probability += prob_arr[i]
        if probability >= random_val:
            return i
        else:
            continue


def gen_offspring(idx_list, pop, fit_arr, lyr_num, feat_num, lyr_1_out, lyr_2_out, lyr_3_out):
    '''
    Randomly selection a pair of individuals as parents
    Generating a offspring through column-wise crossover and within-column mutation
    '''
    alpha = 100
    trans_fit_arr = np.exp(alpha * fit_arr[0, :])
    trans_fit_arr = trans_fit_arr / np.sum(trans_fit_arr)

    idx_1 = roulette(trans_fit_arr)
    idx_2 = roulette(trans_fit_arr)
    while idx_1 == idx_2:
        idx_2 = roulette(trans_fit_arr)
    print('parent_idx', idx_1, idx_2)
    parent_1 = pop[idx_1]
    parent_2 = pop[idx_2]
    p_mask_1 = indi2mask(parent_1, feat_num, lyr_1_out, lyr_2_out, lyr_3_out)
    p_mask_2 = indi2mask(parent_2, feat_num, lyr_1_out, lyr_2_out, lyr_3_out)
    off_mask = dict()
    for i in range(lyr_num):
        out_num = p_mask_1[i].shape[0]
        x_idx = random.sample(range(out_num), int(out_num / 2))
        off_mask[i] = copy.deepcopy(p_mask_1[i])
        off_mask[i][x_idx, :] = p_mask_2[i][x_idx, :]

        c = 0.1
        row_mute_prob = c * np.sum(np.abs(p_mask_1[i] - p_mask_2[i]), axis=1) / p_mask_1[i].shape[1]
        mute_row = np.where(row_mute_prob - np.random.rand(row_mute_prob.shape[0], ) > 0)[0]
        print('mute row num:', len(mute_row))
        for mr in mute_row:
            np.random.shuffle(off_mask[i][mr, :])
    print("sparse degree:")
    print('mask_1:', cal_sc_percent(p_mask_1), '%')
    print('mask_2:', cal_sc_percent(p_mask_2), '%')
    print('mask', cal_sc_percent(off_mask), '%')
    offspring = mask2indi(off_mask)
    return offspring


def comb_selection(pop, fit_arr, offspring, off_fit_arr, pop_size):
    '''
    Adding the offspring to the population
    Update the population through 2-tournament selection
    '''
    pop[len(pop.keys())] = offspring
    fit_arr = np.concatenate((fit_arr, off_fit_arr), axis=1)

    elite_fitness = np.max(fit_arr[0, :])
    elite_no = np.where(fit_arr[0, :] == elite_fitness)[0][0]
    elite = pop[elite_no]
    elite_tacc = fit_arr[1, :][elite_no]

    unselected_idx = np.arange(fit_arr.shape[1])

    for i in range(pop_size):
        idx_0, idx_1 = np.random.choice(unselected_idx, 2, replace=False)
        if fit_arr[0, idx_0] > fit_arr[0, idx_1]:
            unselected_idx = np.delete(unselected_idx, np.where(unselected_idx == idx_0)[0])
        else:
            unselected_idx = np.delete(unselected_idx, np.where(unselected_idx == idx_1)[0])

    # Elite reserving
    if elite_no in unselected_idx:
        unselected_idx = np.delete(unselected_idx, np.where(unselected_idx == elite_no)[0])

    for i in unselected_idx:
        pop.pop(i)

    new_pop = dict()
    for i in range(len(pop.keys())):
        new_pop[i] = pop[list(pop.keys())[i]]
    fit_arr = np.delete(fit_arr, unselected_idx, axis=1)

    print('unselected_idx: ', unselected_idx)
    print(fit_arr)
    print('elite fitness tacc: ', elite_fitness, elite_tacc)

    return new_pop, fit_arr, elite, elite_fitness, elite_tacc


if __name__ == '__main__':
    feat_num = feat_mat.shape[1]
    # Choose an architecture
    # [200]-[100] -------------------------------------------
    # lyr_1_out = 800
    # lyr_2_out = 400
    # lyr_3_out = 2

    # [400]-[200] -------------------------------------------
    # lyr_1_out = 800
    # lyr_2_out = 400
    # lyr_3_out = 2

    # [800]-[400] -------------------------------------------
    lyr_1_out = 800
    lyr_2_out = 400
    lyr_3_out = 2

    # [400]-[200]-[100] -------------------------------------
    # lyr_1_out = 400
    # lyr_2_out = 200
    # lyr_3_out = 100

    # [800]-[400]-[200]--------------------------------------
    # lyr_1_out = 800
    # lyr_2_out = 400
    # lyr_3_out = 200

    # [1600]-[800]-[400]-------------------------------------
    # lyr_1_out = 1600
    # lyr_2_out = 800
    # lyr_3_out = 400

    max_epoch = 2000
    dnn_param = [lyr_1_out, lyr_2_out, lyr_3_out, max_epoch]

    lyr_num = 3 if lyr_3_out == 2 else 4
    init_sizes = [10, 3, 3] if lyr_num == 3 else [5, 3, 3, 2]
    sparsity_range = np.array([[0.01, 0.9, 0.9], [0.05, 1, 1]] if lyr_num == 3 else
                              [[0.01, 0.9, 0.9, 0.9], [0.05, 1, 1, 1]])
    beta = 1e3
    max_gen = 10000

    # Calculate init_pop_size
    init_pop_size = 1
    for i in range(len(init_sizes)):
        init_pop_size = init_pop_size * init_sizes[i]
    print('init_pop_size: ', init_pop_size)

    # generate connections & idx_list -------------------------------------------------------------------------------
    connections = dict()
    idx_list = dict()
    i = 0
    for in_dim, out_dim in [(feat_num, lyr_1_out),
                            (lyr_1_out, lyr_2_out),
                            (lyr_2_out, lyr_3_out),
                            (lyr_3_out, 2)]:
        connections[i], idx_list[i] = gen_conn(in_dim, out_dim)
        i += 1
        if lyr_3_out == 2 and i == 3:
            break

    # EA
    # Initialization:---------------------------------------------------------------------------------------------------
    set_random_seed(0)

    log_path = f'./ea_log'
    if not os.path.exists(log_path):
        os.mkdir(log_path)
    mask_path = './mask'
    if not os.path.exists(mask_path):
        os.mkdir(mask_path)

    # Generating or loading the initial population and evaluate fitness
    if os.path.exists(f'{log_path}/fit_arr_0.npy'):
        init_pop = np.load(f'{log_path}/pop_0.npy', allow_pickle=True).item()
        init_fit_arr = np.load(f'{log_path}/fit_arr_0.npy', allow_pickle=True)
        init_fit_arr = np.floor(init_fit_arr * 1e6) / 1e6
    else:
        # Initial population
        init_pop = gen_init_pop(init_sizes, idx_list, sparsity_range, lyr_num)

        # Initial fitness array
        init_fit_arr = cal_pop_fitness(init_pop, feat_num, lyr_1_out, lyr_2_out, lyr_3_out, dnn_param)
        init_fit_arr = np.floor(init_fit_arr * 1e6) / 1e6
        elite_fitness = np.max(init_fit_arr[0, :])
        elite_tacc = init_fit_arr[1, :][np.where(init_fit_arr[0, :] == elite_fitness)[0][0]]

        np.save(f'{log_path}/pop_0', init_pop)
        np.save(f'{log_path}/fit_arr_0', init_fit_arr)
        np.save(f'{log_path}/0_{"%.4f" % elite_fitness}_{"%.4f" % elite_tacc}', 0)  # 用于观察

    pop = init_pop
    fit_arr = init_fit_arr

    pop_size_cnt = np.array([init_pop_size, 0])

    # Calculating the population size of the next generation using PSM
    pop_size = psm(init_pop_size, init_fit_arr, fit_arr, pop_size_cnt, beta)

    # Evolve start -----------------------------------------------------------------------------------------------------
    save_dnn = 0
    for gen in range(1, max_gen):
        print('gen: ', gen)

        time_s = time.time()

        set_random_seed(gen)

        offspring = gen_offspring(idx_list, pop, fit_arr, lyr_num, feat_num, lyr_1_out, lyr_2_out, lyr_3_out)
        off_fit_arr = cal_indi_fitness(offspring, feat_num, lyr_1_out, lyr_2_out, lyr_3_out, dnn_param, save_dnn=0)
        off_fit_arr = np.floor(off_fit_arr * 1e6) / 1e6
        pop, fit_arr, elite, elite_fitness, elite_tacc = comb_selection(pop, fit_arr, offspring, off_fit_arr, pop_size)
        np.save(f'{log_path}/latest_offspring', offspring)
        np.save(f'{log_path}/off_fit_arr_{gen}', off_fit_arr)
        np.save(f'{log_path}/latest_pop', pop)
        np.save(f'{log_path}/elite', elite)
        np.save(f'{mask_path}/elite_mask_{lyr_1_out}_{lyr_2_out}_{lyr_3_out}',
                indi2mask(elite, feat_num, lyr_1_out, lyr_2_out, lyr_3_out))
        np.save(f'{log_path}/fit_arr_{gen}', fit_arr)
        np.save(f'{log_path}/{gen}_{"%.4f" % elite_fitness}_{"%.4f" % elite_tacc}', 0)  # For easy observation

        pop_size = psm(init_pop_size, init_fit_arr, fit_arr, pop_size_cnt, beta)

        if pop_size == 1:
            cal_indi_fitness(elite, feat_num, lyr_1_out, lyr_2_out, lyr_3_out, dnn_param, save_dnn=0)
            break

        time_e = time.time()
        print('gen time: ', time_e - time_s)
        print('-' * 50)
