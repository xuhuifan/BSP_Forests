import numpy as np
# import autograd.numpy as np
import scipy.io
from scipy.spatial import ConvexHull
from scipy.stats import t, invgamma
# import matplotlib.pyplot as plt
import pandas as pd
# import autograd as grad
from sklearn import linear_model


# global likelihood_setting


def pre_process_data(xdata, ydata, train_test_ratio):
    test_index = np.random.choice(len(ydata), int(np.ceil(len(ydata)*train_test_ratio)), replace=False)
    train_index = np.delete(np.arange(len(ydata)), test_index)

    xdata_train = xdata[train_index]
    ydata_train = ydata[train_index]
    xdata_test = xdata[test_index]
    ydata_test = ydata[test_index]

    ydata_train_min = np.min(ydata_train)
    ydata_train_max = np.max(ydata_train)
    ydata_train_mean = (ydata_train_max+ydata_train_min)/2
    dd = ydata_train_max-ydata_train_min

    ydata_train = (ydata_train-ydata_train_mean)/dd
    ydata_test = (ydata_test-ydata_train_mean)/dd


    # set up the hyper-parameters
    regr = linear_model.LinearRegression()
    regr.fit(xdata_train, ydata_train)
    y_train_predict = regr.predict(xdata_train)
    variance_hat = np.var(y_train_predict-ydata_train)

    hyper_sigma_1 = 1.5
    percentile_val = 0.9

    val1 = invgamma.ppf(percentile_val, a = hyper_sigma_1, scale=1)
    hyper_sigma_2 = variance_hat/val1
    # invgamma.cdf(variance_hat, a=hyper_sigma_1, scale=hyper_sigma_2)
    # Calculate the standard deviation for least square regression

    return xdata_train, ydata_train, xdata_test, ydata_test, ydata_train_mean, dd, hyper_sigma_1, hyper_sigma_2, variance_hat


def line(p1, p2):
    A = (p1[1] - p2[1])
    B = (p2[0] - p1[0])
    C = (p1[0]*p2[1] - p2[0]*p1[1])
    return A, B, -C

def intersection(L1, L2):
    D  = L1[0] * L2[1] - L1[1] * L2[0]
    Dx = L1[2] * L2[1] - L1[1] * L2[2]
    Dy = L1[0] * L2[2] - L1[2] * L2[0]
    if D != 0:
        x = Dx / D
        y = Dy / D
        return [x,y]
    else:
        return False

def block_projection(xdata, z_label_i, label_i, dim_pair):
    dist_i_seq = []
    points_i_seq = []
    label_i_indicator = (z_label_i == label_i)
    if np.sum(label_i_indicator) > 2:
        label_i_x_full = xdata[label_i_indicator]
        for dim_pair_i in dim_pair:
            label_i_x = label_i_x_full[:, dim_pair_i]

            label_i_points = label_i_x[ConvexHull(label_i_x).vertices]
            back_points = np.append(label_i_points[1:], label_i_points[0].reshape((1, -1)), axis=0)
            dist_i = np.sqrt(np.sum((label_i_points - back_points) ** 2, axis=1))

            dist_i_seq.append(dist_i.sum())
            points_i_seq.append(label_i_points)
    return dist_i_seq, points_i_seq

def BSP_forest_projection(xdata, dim_pair, z_label_i):
    cu_canset = np.unique(z_label_i)
    dist_seq = []
    points_seq = []
    for label_i in cu_canset:
        dist_i_seq, points_i_seq = block_projection(xdata, z_label_i, label_i, dim_pair)
        dist_seq.append(dist_i_seq)
        points_seq.append(points_i_seq)
    return dist_seq, points_seq

def sequentialProcess_gen(sequentialPoints):


    rs_correct = 1
    while (rs_correct == 1):
        orthogonal_theta = np.random.rand()*np.pi
        positionB = np.array([np.cos(orthogonal_theta), np.sin(orthogonal_theta)])
        sequential_dis = np.sum(sequentialPoints*positionB, axis = 1)
        seq_max = np.max(sequential_dis)
        seq_min = np.min(sequential_dis)
        largeRatio = np.pi *np.sqrt(2)

        if (largeRatio*np.random.rand() <= (seq_max-seq_min)):
            rs_correct = 0
    cut_position = seq_min*positionB+np.random.rand()*(seq_max-seq_min)*positionB
    cut_direction = np.array([-positionB[1], positionB[0]])

    point3 = cut_position
    point4 = cut_position + cut_direction

    return np.vstack((point3, point4))
    # intersects_two = []
    # intersects_index = []
    #
    # for ii in range(len(sequentialPoints)):
    #     point1 = sequentialPoints[ii]
    #     if ii == (len(sequentialPoints)-1):
    #         point2 = sequentialPoints[0]
    #     else:
    #         point2 = sequentialPoints[ii+1]
    #
    #     intersection_point = intersection(line(point1, point2), line(point3, point4))
    #     if np.sum((intersection_point-point1)*(intersection_point-point2))<0:
    #         intersects_two.append(intersection_point)
    #         intersects_index.append(ii)
    #
    # return np.asarray(intersects_two), intersects_index


def perimeter_cal(selected_points):
    d1 = np.sum(np.sum((selected_points[:-1]-selected_points[1:])**2, axis=1)**(0.5))
    d2 = (np.sum((selected_points[0]-selected_points[-1])**2)**(0.5))
    return d1 + d2

def distcal(selected_points):
    return (np.sum((selected_points[0]-selected_points[1])**2)**(0.5))


def negative_log_pdf_fun(mus, other_val, ydata_sub):

    likelihood_case = 1
    if likelihood_case == 1:
        total_mu_vec = mus+other_val

        super_mu_vec = np.sum(total_mu_vec[ydata_sub==0])
        numerator_log = (-super_mu_vec)
        denominator_log = np.sum(np.log(1+np.exp(-total_mu_vec)))

    return -numerator_log+denominator_log

# def Hamitonian_MC_Sampler(leapfrog_stepsize, leapfrog_num, mus, other_val, ydata_sub):
#     p_gradient = grad(negative_log_pdf_fun)
#     pp = np.random.rand()
#
#     pp = pp - leapfrog_stepsize*p_gradient(mus, other_val, ydata_sub)/2
#     for jj in range(leapfrog_num):
#         mus = mus + leapfrog_stepsize*pp
#         if jj <(leapfrog_num-1):
#             pp = pp - leapfrog_stepsize*p_gradient(mus, other_val, ydata_sub)/2
#     pp = pp - leapfrog_stepsize*p_gradient(mus, other_val, ydata_sub)/2
#
#     return mus, pp


def synthetic_data_gen(dataNum, dimNum, x_range, likelihood_setting, noise_level):
    noiseNess = [0.3, 0.7]
    if dimNum == 1:
        # print x_range
        # xdata = np.arange(x_range[0], x_range[1], (x_range[1]-x_range[0])/dataNum)
        xdata = np.arange(x_range[0], x_range[1], 0.01)
        if noise_level<2:
            noisess = np.random.randn(dataNum)*noiseNess[noise_level]

        if likelihood_setting == 0:
            ydata = xdata**3 + noisess
        elif likelihood_setting == 1:
            ydata = np.sin(8*np.pi*xdata) + noisess
        elif likelihood_setting == 2:
            ydata = np.ones(len(xdata))
            ydata[xdata<0] = -1
            ydata = ydata + noisess
        elif likelihood_setting == 3: # assume label number is 2
            inteception_points = np.sort(np.random.uniform(low=-0.99, high = 0.99, size = 7))
            ydata = np.ones(len(xdata))
            ydata[(inteception_points[0]<=xdata)&(xdata<inteception_points[1])] = 0
            ydata[(inteception_points[2]<=xdata)&(xdata<inteception_points[3])] = 0
            ydata[(inteception_points[4]<=xdata)&(xdata<inteception_points[5])] = 0
            ydata[inteception_points[6]<=xdata] = 0
    else:
        xdata = np.random.uniform(low = x_range[0],high = x_range[1], size = [dataNum, dimNum])
        ydata = np.sum(xdata**3, axis = 1) + np.random.randn(dataNum)*noiseNess
    return ydata, xdata

def tree_stage_cut(dts_star, budget_val):

    cumsum_cost = np.cumsum(dts_star.costseq)
    stage_i = np.sum(cumsum_cost<budget_val)+1

    dts_star.treev = dts_star.treev[:(dts_star.nodeNumseq[stage_i]+1), :]

    condition_judge = (dts_star.treev[:, 1] > dts_star.treev[-1, 0])
    dts_star.treev[condition_judge, 1] = 0
    dts_star.treev[condition_judge, 2] = 0

    dts_star.mu_variance_seq = dts_star.mu_variance_seq[:(dts_star.nodeNumseq[stage_i]+1)]

    dts_star.budgetseq = dts_star.budgetseq[:(stage_i+1)]

    dts_star.costseq = dts_star.costseq[:stage_i]
    dts_star.propose_ratio = dts_star.propose_ratio[:stage_i]
    dts_star.z_label = dts_star.z_label[:(stage_i+1)]
    dts_star.nodeNumseq = dts_star.nodeNumseq[:(stage_i+1)]

    dts_star.cut_block_seq = dts_star.cut_block_seq[:stage_i]

    dts_star.cut_line_points = dts_star.cut_line_points[:stage_i]
    dts_star.dim_pair_index = dts_star.dim_pair_index[:stage_i]

    dts_star.dists_seq = dts_star.dists_seq[:(dts_star.nodeNumseq[stage_i]+1)]
    dts_star.points_seq = dts_star.points_seq[:(dts_star.nodeNumseq[stage_i]+1)]

    return dts_star




def non_increasing(L):
    return all(x>=y for x, y in zip(L, L[1:]))

def non_decreasing(L):
    return all(x<=y for x, y in zip(L, L[1:]))

def index_judge(current_lower, current_upper, xdata):
    if xdata.ndim==1:
        return np.where(((current_lower <= xdata) * (current_upper > xdata)).astype(bool))
    else:
        return np.where((np.prod(current_lower<=xdata, axis=1)*np.prod(current_upper>xdata, axis=1)).astype(bool))


def auto_corr(M):
#   The autocorrelation has to be truncated at some point so there are enough
#   data points constructing each lag. Let kappa be the cutoff
    kappas = len(M)
    auto_corrval = np.zeros(kappas-30)
    mu = np.mean(M)
    for s in range(1,kappas-30):
        # auto_corrval[s] = np.corrcoef(M[:-s],M[s:])[0, 1]
        # auto_corrval[s] = np.sum((M[:-s]-mu)*(M[s:]-mu))/np.sum((M-mu)*(M-mu))
        auto_corrval[s] = np.mean((M[:-s]-mu) * (M[s:]-mu)) / np.var(M)
    return auto_corrval, 1+2*np.sum(auto_corrval)

# def auto_corr_1(M):
# #   The autocorrelation has to be truncated at some point so there are enough
# #   data points constructing each lag. Let kappa be the cutoff
#     kappas = len(M)
#     auto_corrval = np.zeros(kappas-30)
#     mu = np.mean(M)
#     for s in range(1,kappas-30):
#         auto_corrval[s] = np.corrcoef(M[:-s],M[s:])[0, 1]
#         # auto_corrval[s] = np.mean( (M[:-s]-mu) * (M[s:]-mu) ) / np.var(M)
#     return auto_corrval, 1+2*np.sum(auto_corrval)
#
# def auto_corr_2(M):
# #   The autocorrelation has to be truncated at some point so there are enough
# #   data points constructing each lag. Let kappa be the cutoff
#     kappas = len(M)
#     auto_corrval = np.zeros(kappas-30)
#     mu = np.mean(M)
#     for s in range(1,kappas-30):
#         covval = np.cov(np.vstack((M[:-s],M[s:])))
#         varval1 = np.var(M[:-s],M[s:])
#         varval2 = np.cov(M[:-s],M[s:])
#         auto_corrval[s] = covval[0, 1]/((varval1**(0.5))*(varval2**(0.5)))
#     return auto_corrval, 1+2*np.sum(auto_corrval)


def read_data():
    datafile = 'boston/boston_data.csv'
    vals = pd.read_csv(datafile, header = 0)
    xydata = vals.values
    xyxy = xydata[np.arange(1, len(xydata), 2)]
    ydata = xyxy[:, -1]
    xdata = xyxy[:, :(-1)]
    return ydata, xdata

def autocorr(x):
    result = np.correlate(x, x, mode='full')
    return result[np.ceil(result.size/2.0).astype(int):]