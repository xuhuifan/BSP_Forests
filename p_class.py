import numpy as np
import scipy.io
from scipy.stats import norm
from scipy.stats import truncnorm
from scipy.stats import t, uniform
from scipy.stats import invgamma
# import matplotlib.pyplot as plt
import copy

from utility import *



def Friedman_function_gen(dimNum,dataNum):

    xdata = uniform.rvs(size=(dataNum, dimNum))
    ydata = 10*np.sin(xdata[:, 0]*np.pi*xdata[:, 1])+20*((xdata[:, 2]-0.5)**2)+10*xdata[:, 3]+5*xdata[:, 4]+norm.rvs(size=dataNum)

    return xdata, ydata




def BSP_syn_data_random_generator(dimNum, dataNum, budget):

    xdata = np.random.rand(dataNum, dimNum)

    ydata = np.random.rand(dataNum)
    add_dts_other_mu = np.zeros(dataNum)
    mus = 0
    variances = 1
    mTree = 10
    maxStage = 100

    random_choice = 1

    ydata_total = np.zeros(dataNum)

    for rr in range(random_choice):

        # dims_sele = np.random.choice(dimNum, 2, replace=False)
        # dims_sele = np.array([0, 1])

        NumPossbile = int(dimNum*(dimNum-1)/2)
        alpha_val = 10

        dim_weight = np.random.dirichlet(alpha=np.ones(NumPossbile)*alpha_val/(NumPossbile))
        pari_sele = np.random.choice(len(dim_weight), p=dim_weight)
        dim_pair = []
        for k1 in range(dimNum-1):
            for k2 in np.arange(k1+1, dimNum):
                dim_pair.append([k1, k2])
        data_dts = dts(mus, variances, budget, mTree, ydata, xdata, dim_pair)

        # for tt in np.arange(maxStage):
        #     data_dts.propose_cut(xdata, ydata, add_dts_other_mu, add_dts_other_variance)
        #     # if data_dts.budgetseq[-1] < 0:
        #     break

        # plt.scatter(xdata[data_dts.z_label[-1] == 1, 0], xdata[data_dts.z_label[-1]== 1, 1], color='red')
        # plt.scatter(xdata[data_dts.z_label[-1]== 2, 0], xdata[data_dts.z_label[-1]== 2, 1], color='green')
        # plt.show()

        data_dts.tree_full_gen(maxStage, xdata, ydata, add_dts_other_mu)

        label_unique = np.unique(data_dts.z_label[-1])
        min_u = -10
        max_u = 10
        # if len(label_unique)==0:
        #     a = 1
        # print(label_unique)
        # muss = np.arange(min_u, max_u, (max_u-min_u)/len(label_unique))
        muss = np.random.uniform(min_u, max_u, size=len(label_unique))
        # muss = np.random.normal(loc = 0,scale = 10, size=len(label_unique))
        ydata = np.zeros(dataNum)
        for [label_i, mu_i] in zip(label_unique, muss):
            ydata[data_dts.z_label[-1]==label_i] = (np.random.normal(loc = mu_i, scale = 0.1, size=np.sum(data_dts.z_label[-1]==label_i)))
        ydata_total = ydata_total + ydata


    return xdata, ydata_total, dimNum, dataNum



class dts:
    def __init__(self, mus, true_var, budget, mTree, ydata, xdata, dim_pair):

        # treev: 0, parent node index 1, left subtree node index 2, right subtree node index
        self.treev = np.zeros((1, 3), dtype=int)

        self.mus = mus
        self.true_var = true_var

        self.mu_variance_seq = np.array([mus])
        self.mTree = mTree

        self.budgetseq = [budget]

        self.costseq = []
        self.propose_ratio = []
        self.z_label = [np.zeros(len(ydata), dtype=int)]

        self.dim_pair = dim_pair
        self.nodeNumseq = [0]

        self.cut_line_points = []
        self.dim_pair_index = []

        dists_seq, points_seq = BSP_forest_projection(xdata, self.dim_pair, self.z_label[-1])
        self.dists_seq = dists_seq
        self.points_seq = points_seq

        self.cut_block_seq = []



    def propose_cut(self, xdata, ydata, add_dts_other_mu, cu_canset, block_wise_proportion):

        # cu_canset = self.treev[self.treev[:, 1]==0, 0]

        # cu_canset, cu_canset_counts = np.unique(self.z_label[-1], return_counts=True)
        # cu_canset = cu_canset[cu_canset_counts > 5]


        block_select = np.random.choice(cu_canset, p=block_wise_proportion/np.sum(block_wise_proportion))

        self.cut_block_seq.append(block_select)
        block_select_perimeters = self.dists_seq[block_select]

        dim_pair_select = np.random.choice(len(self.dim_pair), p = block_select_perimeters/np.sum(block_select_perimeters))
        sequentialPoints = self.points_seq[block_select][dim_pair_select]

        c_p = sequentialProcess_gen(sequentialPoints)

        self.cut_line_points.append(c_p)
        self.dim_pair_index.append(dim_pair_select)

        block_select_index = np.where(self.z_label[-1]==block_select)[0]
        kx = xdata[block_select_index][:,self.dim_pair[dim_pair_select]]
        index_p = (((kx[:, 0]-c_p[0, 0])*(c_p[1, 1]-c_p[0, 1])-(kx[:, 1]-c_p[0, 1])*(c_p[1, 0]-c_p[0, 0]))<0)

        propose_z = copy.deepcopy(self.z_label[-1])
        k1_index = block_select_index[index_p]
        k2_index = block_select_index[~index_p]
        propose_z[k1_index] = self.nodeNumseq[-1]+1
        propose_z[k2_index] = self.nodeNumseq[-1]+2
        self.z_label.append(propose_z)

        self.treev[block_select, 1] = self.nodeNumseq[-1] + 1
        self.treev[block_select, 2] = self.nodeNumseq[-1] + 2

        add_treev = np.zeros((2, 3), dtype=int)
        add_treev[0, 0] = self.nodeNumseq[-1] + 1
        add_treev[1, 0] = self.nodeNumseq[-1] + 2

        self.treev = np.vstack((self.treev, add_treev))


        for label_i in [self.nodeNumseq[-1]+1, self.nodeNumseq[-1]+2]:
            dist_i_seq, points_i_seq = block_projection(xdata, self.z_label[-1], label_i, self.dim_pair)
            self.dists_seq.append(dist_i_seq)
            self.points_seq.append(points_i_seq)

        # scale_val = len(self.dim_pair)
        # self.costseq.append(np.random.exponential(scale=scale_val/np.sum(block_wise_proportion)))
        # self.budgetseq.append(self.budgetseq[-1]-self.costseq[-1])



        # sample the mu and sigma value in the newly generated node

        y_differ = ydata - add_dts_other_mu
        y_differ_1 = y_differ[k1_index]
        y_differ_2 = y_differ[k2_index]

        ###################
        # the specification here is quite important for the final result
        pre_var_1 = self.true_var
        pre_var_2 = self.true_var

        prior_mu = self.mus
        prior_var = ((0.5 / 3) ** 2)/self.mTree

        # prior_var = ((np.max(ydata)-np.min(ydata))**2)/(4*4*self.mTree)

        posterior_var_1 = (prior_var ** (-1) + len(y_differ_1) / pre_var_1) ** (-1)
        posterior_mu_1 = posterior_var_1 * (prior_mu / prior_var + np.sum(y_differ_1) /pre_var_1)
        mu1 = norm.rvs(posterior_mu_1, posterior_var_1 ** (0.5))

        posterior_var_2 = (prior_var ** (-1) + len(y_differ_2) / pre_var_2) ** (-1)
        posterior_mu_2 = posterior_var_2 * (prior_mu / prior_var + np.sum(y_differ_2) /pre_var_2)
        mu2 = norm.rvs(posterior_mu_2, posterior_var_2 ** (0.5))

        returned_ratio_mu1 = norm.logpdf(mu1, loc = prior_mu, scale = (prior_var)**(0.5))-norm.logpdf(mu1, loc=posterior_mu_1, scale=(posterior_var_1)**(0.5))
        returned_ratio_mu2 = norm.logpdf(mu2, loc = prior_mu, scale = (prior_var)**(0.5))-norm.logpdf(mu2, loc=posterior_mu_2, scale=(posterior_var_2)**(0.5))

        self.mu_variance_seq = np.append(self.mu_variance_seq, [mu1, mu2])

        self.nodeNumseq.append(self.nodeNumseq[-1]+2)
        self.propose_ratio.append(returned_ratio_mu1+returned_ratio_mu2)

    def tree_full_gen(self, maxStage, xdata, ydata, add_dts_other_mu):

        totalBudget = self.budgetseq[0]
        budget_step = totalBudget/maxStage
        budgetSeq = np.arange(0, totalBudget+budget_step, budget_step)
        for max_i in range(maxStage):

                # We will only take the partition one cut back
                while np.sum(self.costseq)<budgetSeq[max_i+1]:
                    cu_canset, cu_canset_counts = np.unique(self.z_label[-1], return_counts=True)
                    nonempty_index = (cu_canset_counts>5)

                    if np.sum(nonempty_index)>0:
                        cu_canset = cu_canset[nonempty_index]
                        block_wise_proportion = [sum(self.dists_seq[dists_seq_i]) for dists_seq_i in cu_canset]
                        scale_val = len(self.dim_pair)
                        self.costseq.append(np.random.exponential(scale=scale_val / np.sum(block_wise_proportion)))
                        self.budgetseq.append(self.budgetseq[-1] - self.costseq[-1])

                    else: # This case refer to the case of cost is infinity and will not be used. Only for index convenience
                        block_wise_proportion = [sum(self.dists_seq[dists_seq_i]) for dists_seq_i in cu_canset]
                        self.costseq.append(np.inf)
                        self.budgetseq.append(-np.inf)

                    self.propose_cut(xdata, ydata, add_dts_other_mu, cu_canset, block_wise_proportion)


    def predict_data(self, xdata_test):

        temp_label = np.zeros(len(xdata_test),dtype=int)
        for ii in range(len(self.cut_line_points)):
            c_p = self.cut_line_points[ii]
            ii_index = np.where(temp_label==self.cut_block_seq[ii])[0]
            kx = xdata_test[ii_index][:,self.dim_pair[self.dim_pair_index[ii]]]
            index_p = (((kx[:, 0]-c_p[0, 0])*(c_p[1, 1]-c_p[0, 1])-(kx[:, 1]-c_p[0, 1])*(c_p[1, 0]-c_p[0, 0]))<0)
            temp_label[ii_index[index_p]] = self.treev[self.cut_block_seq[ii], 1]
            temp_label[ii_index[~index_p]] = self.treev[self.cut_block_seq[ii], 2]

        predict_y_test = self.mu_variance_seq[temp_label]
        return predict_y_test


    def assign_to_data(self, stage_i):
        terminal_node = self.z_label[stage_i]
        mi_dts_mu_data = self.mu_variance_seq[terminal_node]

        return mi_dts_mu_data

    def ll_cal(self, ydata, add_dts_other_mu, stage_i):
        # we need to define the likelihood here for cutting the block
        mi_dts_mu_data = self.assign_to_data(stage_i)
        predicted_y_mu = add_dts_other_mu+mi_dts_mu_data

        return np.sum(norm.logpdf(ydata, predicted_y_mu, self.true_var**(0.5)))




    def dts_update(self, particleNUm, dts_star, maxStage, xdata, ydata, add_dts_other_mu):
        par_dts_seq = []

        for pari in range(particleNUm):
            dts_pari = dts(self.mus, self.true_var, self.budgetseq[0], self.mTree, ydata, xdata, self.dim_pair)
            par_dts_seq.append(dts_pari)

        currents_ll = np.zeros(particleNUm + 1)
        previous_ll = np.zeros(particleNUm + 1)

        totalBudget = self.budgetseq[0]
        budget_step = totalBudget/maxStage
        budgetSeq = np.arange(0, totalBudget+budget_step, budget_step)
        for max_i in range(maxStage):


            for pari in range(particleNUm):

                # We will only take the partition one cut back
                while np.sum(par_dts_seq[pari].costseq)<budgetSeq[max_i+1]:
                    cu_canset, cu_canset_counts = np.unique(par_dts_seq[pari].z_label[-1], return_counts=True)
                    nonempty_index = (cu_canset_counts>5)

                    if np.sum(nonempty_index)>0:
                        cu_canset = cu_canset[nonempty_index]
                        block_wise_proportion = [sum(par_dts_seq[pari].dists_seq[dists_seq_i]) for dists_seq_i in cu_canset]
                        scale_val = len(par_dts_seq[pari].dim_pair)
                        par_dts_seq[pari].costseq.append(np.random.exponential(scale=scale_val / np.sum(block_wise_proportion)))
                        par_dts_seq[pari].budgetseq.append(par_dts_seq[pari].budgetseq[-1] - par_dts_seq[pari].costseq[-1])

                    else: # This case refer to the case of cost is infinity and will not be used. Only for index convenience
                        block_wise_proportion = [sum(par_dts_seq[pari].dists_seq[dists_seq_i]) for dists_seq_i in cu_canset]
                        par_dts_seq[pari].costseq.append(np.inf)
                        par_dts_seq[pari].budgetseq.append(-np.inf)

                    par_dts_seq[pari].propose_cut(xdata, ydata, add_dts_other_mu, cu_canset, block_wise_proportion)

            # assign dts_star to the last element of par_dts_seq
            copy_dts_star = copy.deepcopy(dts_star)
            # if np.sum(copy_dts_star.costseq)<totalBudget:
            #     a = 1
            par_dts_seq.append(tree_stage_cut(copy_dts_star, budgetSeq[max_i+1]))

            # calculate the log-likelihood sequence
            ll_only = np.zeros(particleNUm+1)
            for pari in range(particleNUm+1):
                cumsum_cost = np.cumsum(par_dts_seq[pari].costseq)
                max_i_index = (budgetSeq[max_i] < cumsum_cost) & (budgetSeq[max_i + 1] >= cumsum_cost)
                propose_ratio_max_i = np.sum(np.asarray(par_dts_seq[pari].propose_ratio)[max_i_index])
                ll_only[pari] = (par_dts_seq[pari].ll_cal(ydata, add_dts_other_mu, len(par_dts_seq[pari].costseq) - 1))
                currents_ll[pari] = currents_ll[pari]+propose_ratio_max_i+ll_only[pari]

            ll_seqi_ratio = currents_ll-previous_ll
            propb = np.exp(ll_seqi_ratio-np.max(ll_seqi_ratio))

            if (np.sum(propb**2)*(particleNUm+1))<(0.5*((np.sum(propb))**2)):
                select_index = np.random.choice((particleNUm + 1), particleNUm, replace=True, p=propb / np.sum(propb))
                par_dts_seq = [copy.deepcopy(par_dts_seq[i]) for i in select_index]

                previous_ll[:particleNUm] = currents_ll[select_index]
                previous_ll[-1] = currents_ll[-1]
            else:
                currents_ll = currents_ll - ll_only
                if max_i!=(maxStage-1):
                    par_dts_seq.pop()

        select_index = np.random.choice((particleNUm + 1), p=propb / np.sum(propb))
        final_particle = par_dts_seq[select_index]



        self.treev = final_particle.treev

        self.mu_variance_seq = final_particle.mu_variance_seq

        self.budgetseq = final_particle.budgetseq

        self.costseq = final_particle.costseq
        self.propose_ratio = final_particle.propose_ratio
        self.z_label = final_particle.z_label
        self.cut_block_seq = final_particle.cut_block_seq

        self.nodeNumseq = final_particle.nodeNumseq

        self.cut_line_points = final_particle.cut_line_points
        self.dim_pair_index = final_particle.dim_pair_index
        self.dists_seq = final_particle.dists_seq
        self.points_seq = final_particle.points_seq



class add_dts:

    def __init__(self, mTree, mus, variances, maxStage, budget, xdata, ydata, dimNum):
        self.mTree = mTree
        self.true_var = variances

        add_dts = []
        add_dts_other_mu = np.zeros(len(ydata))

        dim_pair = []
        for k1 in range(dimNum-1):
            for k2 in np.arange(k1+1, dimNum):
                dim_pair.append([k1, k2])
        self.dim_pair = dim_pair

        for mi in range(mTree):

            mi_dts = dts(mus, self.true_var, budget, mTree, ydata, xdata, self.dim_pair)
            mi_dts.tree_full_gen(maxStage, xdata, ydata, add_dts_other_mu)
            add_dts.append(mi_dts)

            add_dts_mi_mu = mi_dts.assign_to_data(int(mi_dts.nodeNumseq[-1]/2))
            add_dts_other_mu = add_dts_other_mu + add_dts_mi_mu
        self.dimNum = dimNum
        self.add_dts = add_dts


    def updates(self, particleNum, maxStage, xdata, ydata, dimNum, hyper_sigma_1, hyper_sigma_2):
        total_mu_mat = np.zeros((self.mTree, len(ydata)))
        for mi in range(self.mTree):
            total_mu_mat[mi] = self.add_dts[mi].assign_to_data(len(self.add_dts[mi].z_label)-1)

        for mi in range(self.mTree):
            add_dts_other_mu = np.sum(total_mu_mat, axis=0)-total_mu_mat[mi]
            dts_mi = copy.deepcopy(self.add_dts[mi])

            dts_mi.dts_update(particleNum, dts_mi, maxStage, xdata, ydata, add_dts_other_mu)
            total_mu_mat[mi] = dts_mi.assign_to_data(len(dts_mi.z_label)-1)
            self.add_dts[mi] = dts_mi


        # update the hyper-parameters of the variance

        posterior_hyper_sigma_1 = hyper_sigma_1 + len(xdata)/ 2  # the number refers to the number of terminal nodes
        square_of_errors = np.sum((np.sum(total_mu_mat, axis=0)-ydata)**2)
        posterior_hyper_sigma_2 = hyper_sigma_2 + square_of_errors / 2
        self.true_var = invgamma.rvs(a = posterior_hyper_sigma_1, loc = 0, scale = posterior_hyper_sigma_2)

        # pass the self.true_var to each tree of the addtree
        for add_dts_i in self.add_dts:
            add_dts_i.true_var = self.true_var

