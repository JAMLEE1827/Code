import numpy as np
from scipy.interpolate import RectBivariateSpline
from scipy.ndimage import binary_dilation
from scipy.stats import gaussian_kde
from .trajectory_utils import prediction_output_to_trajectories
#import visualization
from matplotlib import pyplot as plt
import math
import os
import pdb

def haversine_distance(lat1, lon1, lat2, lon2):
    # 将角度转换为弧度
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    # 哈弗赛公式
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2
    c = 2 * np.arcsin(np.sqrt(a))
    km = 6371 * c  # 地球平均半径, 6371km
    return km

def compute_ade(predicted_trajs, gt_traj):
    predicted_trajs[:, :, :, 1] = predicted_trajs[:, :, :, 1] * (58.0 - 55.5) + 55.5
    predicted_trajs[:, :, :, 0] = predicted_trajs[:, :, :, 0] * (13.0 - 10.3) + 10.3
    gt_traj[:, 1] = gt_traj[:, 1] * (58.0 - 55.5) + 55.5
    gt_traj[:, 0] = gt_traj[:, 0] * (13.0 - 10.3) + 10.3
    predicted_trajs = predicted_trajs.squeeze(0)

    error = haversine_distance(predicted_trajs[:, :, 1],
                               predicted_trajs[:, :, 0],
                               gt_traj[:, 1], gt_traj[:, 0])
    # error = np.linalg.norm(predicted_trajs - gt_traj, axis=-1)
    ade = np.mean(error, axis=-1)
    return ade.flatten()


def compute_fde(predicted_trajs, gt_traj):
    predicted_trajs[:, :, :, 1] = predicted_trajs[:, :, :, 1] * (58.0 - 55.5) + 55.5
    predicted_trajs[:, :, :, 0] = predicted_trajs[:, :, :, 0] * (13.0 - 10.3) + 10.3
    gt_traj[:, 1] = gt_traj[:, 1] * (58.0 - 55.5) + 55.5
    gt_traj[:, 0] = gt_traj[:, 0] * (13.0 - 10.3) + 10.3
    predicted_trajs = predicted_trajs.squeeze(0)

    final_error = haversine_distance(predicted_trajs[:, -1, 1],
                               predicted_trajs[:, -1, 0],
                               gt_traj[-1, 1], gt_traj[-1, 0])
    # final_error = np.linalg.norm(predicted_trajs[:, :, -1] - gt_traj[-1], axis=-1)
    return final_error.flatten()


def compute_kde_nll(predicted_trajs, gt_traj):
    kde_ll = 0.
    log_pdf_lower_bound = -20
    num_timesteps = gt_traj.shape[0]
    num_batches = predicted_trajs.shape[0]

    for batch_num in range(num_batches):
        for timestep in range(num_timesteps):
            try:
                kde = gaussian_kde(predicted_trajs[batch_num, :, timestep].T)
                pdf = np.clip(kde.logpdf(gt_traj[timestep].T), a_min=log_pdf_lower_bound, a_max=None)[0]
                kde_ll += pdf / (num_timesteps * num_batches)
            except np.linalg.LinAlgError:
                kde_ll = np.nan

    return -kde_ll


def compute_obs_violations(predicted_trajs, map):
    obs_map = map.data

    interp_obs_map = RectBivariateSpline(range(obs_map.shape[1]),
                                         range(obs_map.shape[0]),
                                         binary_dilation(obs_map.T, iterations=4),
                                         kx=1, ky=1)

    old_shape = predicted_trajs.shape
    pred_trajs_map = map.to_map_points(predicted_trajs.reshape((-1, 2)))

    traj_obs_values = interp_obs_map(pred_trajs_map[:, 0], pred_trajs_map[:, 1], grid=False)
    traj_obs_values = traj_obs_values.reshape((old_shape[0], old_shape[1]))
    num_viol_trajs = np.sum(traj_obs_values.max(axis=1) > 0, dtype=float)

    return num_viol_trajs

def select_best_samples(prediction_dict, futures_dict):
    best_samples = {}
    for t, nodes in prediction_dict.items():
        best_samples[t] = {}
        for node, predictions in nodes.items():
            # 计算每个样本的ADE
            predictions = predictions.squeeze(0)
            future = futures_dict[t][node]
            ade_errors = [np.mean(np.linalg.norm(pred - future, axis=1)) for pred in predictions]

            # 选择ADE最小的样本
            best_sample_index = np.argmin(ade_errors)
            best_samples[t][node] = predictions[best_sample_index, :, :]
    return best_samples

def plot_trajectories_for_timestep(history_dict, best_samples, futures_dict, batch_number, t):
    plt.figure(figsize=(10, 6))
    for node, history_traj in history_dict[t].items():
        best_pred_traj = best_samples[t][node]
        true_traj = futures_dict[t][node]

        # 绘制历史轨迹
        plt.plot(history_traj[:, 0], history_traj[:, 1], lw=1, linestyle='solid', label=f'历史轨迹 - 节点{node}', color='blue')
        # 绘制最佳预测轨迹
        plt.plot(best_pred_traj[:, 0], best_pred_traj[:, 1], lw=1, linestyle='dotted', label=f'预测轨迹 - 节点{node}', color='green')
        # 绘制实际未来轨迹
        plt.plot(true_traj[:, 0], true_traj[:, 1], lw=1, linestyle='solid', label=f'实际轨迹 - 节点{node}', color='red')

    plt.title(f'trajectory per t{t}')
    plt.xlabel('X coords')
    plt.ylabel('Y coords')
    # plt.show()
    save_path = f'Visuals/batch_{batch_number}'
    os.makedirs(save_path, exist_ok=True)  # 创建目录
    plt.savefig(os.path.join(save_path, f'timestep_{t}_image.png'))
    plt.close()

def compute_batch_statistics(batch_number,
                             prediction_output_dict,
                             dt,
                             max_hl,
                             ph,
                             node_type_enum,
                             kde=True,
                             obs=False,
                             map=None,
                             prune_ph_to_future=False,
                             best_of=False):

    (prediction_dict,
     history_dict,
     futures_dict) = prediction_output_to_trajectories(prediction_output_dict,
                                                       dt,
                                                       max_hl,
                                                       ph,
                                                       prune_ph_to_future=prune_ph_to_future)

    # 可视化每个时间步的轨迹
    best_samples = select_best_samples(prediction_dict, futures_dict)
    for t in prediction_dict.keys():
        plot_trajectories_for_timestep(history_dict, best_samples, futures_dict, batch_number, t)

    batch_error_dict = dict()
    for node_type in node_type_enum:
        batch_error_dict[node_type] = {'ade': list(), 'fde': list(), 'kde': list(), 'obs_viols': list()}

    for t in prediction_dict.keys():
        for node in prediction_dict[t].keys():

            ade_errors = compute_ade(prediction_dict[t][node] * 0.01, futures_dict[t][node] * 0.01)
            fde_errors = compute_fde(prediction_dict[t][node] * 0.01, futures_dict[t][node] * 0.01)
            if kde:
                kde_ll = compute_kde_nll(prediction_dict[t][node], futures_dict[t][node])
            else:
                kde_ll = 0
            if obs:
                obs_viols = compute_obs_violations(prediction_dict[t][node], map)
            else:
                obs_viols = 0
            if best_of:
                ade_errors = np.min(ade_errors, keepdims=True)
                fde_errors = np.min(fde_errors, keepdims=True)
                kde_ll = np.min(kde_ll)
            batch_error_dict[node.type]['ade'].extend(list(ade_errors))
            batch_error_dict[node.type]['fde'].extend(list(fde_errors))
            batch_error_dict[node.type]['kde'].extend([kde_ll])
            batch_error_dict[node.type]['obs_viols'].extend([obs_viols])
    return batch_error_dict


# def log_batch_errors(batch_errors_list, log_writer, namespace, curr_iter, bar_plot=[], box_plot=[]):
#     for node_type in batch_errors_list[0].keys():
#         for metric in batch_errors_list[0][node_type].keys():
#             metric_batch_error = []
#             for batch_errors in batch_errors_list:
#                 metric_batch_error.extend(batch_errors[node_type][metric])

#             if len(metric_batch_error) > 0:
#                 log_writer.add_histogram(f"{node_type.name}/{namespace}/{metric}", metric_batch_error, curr_iter)
#                 log_writer.add_scalar(f"{node_type.name}/{namespace}/{metric}_mean", np.mean(metric_batch_error), curr_iter)
#                 log_writer.add_scalar(f"{node_type.name}/{namespace}/{metric}_median", np.median(metric_batch_error), curr_iter)

#                 if metric in bar_plot:
#                     pd = {'dataset': [namespace] * len(metric_batch_error),
#                                   metric: metric_batch_error}
#                     kde_barplot_fig, ax = plt.subplots(figsize=(5, 5))
#                     visualization.visualization_utils.plot_barplots(ax, pd, 'dataset', metric)
#                     log_writer.add_figure(f"{node_type.name}/{namespace}/{metric}_bar_plot", kde_barplot_fig, curr_iter)

#                 if metric in box_plot:
#                     mse_fde_pd = {'dataset': [namespace] * len(metric_batch_error),
#                                   metric: metric_batch_error}
#                     fig, ax = plt.subplots(figsize=(5, 5))
#                     visualization.visualization_utils.plot_boxplots(ax, mse_fde_pd, 'dataset', metric)
#                     log_writer.add_figure(f"{node_type.name}/{namespace}/{metric}_box_plot", fig, curr_iter)


def print_batch_errors(batch_errors_list, namespace, curr_iter):
    for node_type in batch_errors_list[0].keys():
        for metric in batch_errors_list[0][node_type].keys():
            metric_batch_error = []
            for batch_errors in batch_errors_list:
                metric_batch_error.extend(batch_errors[node_type][metric])

            if len(metric_batch_error) > 0:
                print(f"{curr_iter}: {node_type.name}/{namespace}/{metric}_mean", np.mean(metric_batch_error))
                print(f"{curr_iter}: {node_type.name}/{namespace}/{metric}_median", np.median(metric_batch_error))


def batch_pcmd(prediction_output_dict,
               dt,
               max_hl,
               ph,
               node_type_enum,
               kde=True,
               obs=False,
               map=None,
               prune_ph_to_future=False,
               best_of=False):

    (prediction_dict,
     _,
     futures_dict) = prediction_output_to_trajectories(prediction_output_dict,
                                                       dt,
                                                       max_hl,
                                                       ph,
                                                       prune_ph_to_future=prune_ph_to_future)

    batch_error_dict = dict()
    for node_type in node_type_enum:
        batch_error_dict[node_type] =  {'ade': list(), 'fde': list(), 'kde': list(), 'obs_viols': list()}

    for t in prediction_dict.keys():
        for node in prediction_dict[t].keys():
            ade_errors = compute_ade(prediction_dict[t][node], futures_dict[t][node])
            fde_errors = compute_fde(prediction_dict[t][node], futures_dict[t][node])
            if kde:
                kde_ll = compute_kde_nll(prediction_dict[t][node], futures_dict[t][node])
            else:
                kde_ll = 0
            if obs:
                obs_viols = compute_obs_violations(prediction_dict[t][node], map)
            else:
                obs_viols = 0
            if best_of:
                ade_errors = np.min(ade_errors, keepdims=True)
                fde_errors = np.min(fde_errors, keepdims=True)
                kde_ll = np.min(kde_ll)
            batch_error_dict[node.type]['ade'].append(np.array(ade_errors))
            batch_error_dict[node.type]['fde'].append(np.array(fde_errors))

    return batch_error_dict
