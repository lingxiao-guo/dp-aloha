import torch
import numpy as np
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import pickle
import math
import h5py
import argparse
import matplotlib.pyplot as plt
from copy import deepcopy
from tqdm import tqdm
from einops import rearrange

from act.constants import DT
from act.constants import PUPPET_GRIPPER_JOINT_OPEN
from act.act_utils import load_data  # data functions
from act.act_utils import sample_box_pose, sample_insertion_pose  # robot functions
from act.act_utils import compute_dict_mean, set_seed, detach_dict  # helper functions
from act.policy import ACTPolicy, CNNMLPPolicy, DiffusionPolicy
from act.act_utils import put_text, plot_3d_trajectory
from act.awe_entropy import dp_waypoint_selection, dp_entropy_waypoint_selection
from act.detr.models.entropy_utils import KDE
from act.visualize_episodes import save_videos

from act.sim_env import BOX_POSE

import IPython

e = IPython.embed


def main(args):
    set_seed(1)
    # command line parameters
    is_eval = args["eval"]
    is_eval_speed = args["eval_speed"]
    is_label = args["label"]
    ckpt_dir = args["ckpt_dir"]
    policy_class = args["policy_class"]
    onscreen_render = args["onscreen_render"]
    task_name = args["task_name"]
    batch_size_train = args["batch_size"]
    batch_size_val = args["batch_size"]
    num_epochs = args["num_epochs"]
    use_waypoint = args["use_waypoint"]
    constant_waypoint = args["constant_waypoint"]
    if use_waypoint:
        print("Using waypoint")
    if constant_waypoint is not None:
        print(f"Constant waypoint: {constant_waypoint}")

    # get task parameters
    # is_sim = task_name[:4] == 'sim_'
    is_sim = True  # hardcode to True to avoid finding constants from aloha
    if is_sim:
        from constants import SIM_TASK_CONFIGS

        task_config = SIM_TASK_CONFIGS[task_name]
    else:
        from aloha_scripts.constants import TASK_CONFIGS

        task_config = TASK_CONFIGS[task_name]
    dataset_dir = task_config["dataset_dir"]
    num_episodes = task_config["num_episodes"]
    episode_len = task_config["episode_len"]
    camera_names = task_config["camera_names"]

    # fixed parameters
    state_dim = 14
    lr_backbone = 1e-5
    backbone = "resnet18"
    if policy_class == "ACT":
        enc_layers = 4
        dec_layers = 7
        nheads = 8
        policy_config = {
            "lr": args["lr"],
            "num_queries": args["chunk_size"],
            "kl_weight": args["kl_weight"],
            "hidden_dim": args["hidden_dim"],
            "dim_feedforward": args["dim_feedforward"],
            "lr_backbone": lr_backbone,
            "backbone": backbone,
            "enc_layers": enc_layers,
            "dec_layers": dec_layers,
            "nheads": nheads,
            "camera_names": camera_names,
        }
    elif policy_class == "CNNMLP":
        policy_config = {
            "lr": args["lr"],
            "lr_backbone": lr_backbone,
            "backbone": backbone,
            "num_queries": 1,
            "camera_names": camera_names,
        }
    elif policy_class == "DP":
        encoder_config = {
            "lr": args["lr"],
            "lr_backbone": lr_backbone,
            "backbone": backbone,
            "num_queries": 1,
            "camera_names": camera_names,
        }
        policy_config = {
            "cfg": args["diffusion_policy_cfg"],
            "encoder":encoder_config,
            "num_queries": 48,
        }
    else:
        raise NotImplementedError

    config = {
        "num_epochs": num_epochs,
        "ckpt_dir": ckpt_dir,
        "episode_len": episode_len,
        "state_dim": state_dim,
        "lr": args["lr"],
        "policy_class": policy_class,
        "onscreen_render": onscreen_render,
        "policy_config": policy_config,
        "task_name": task_name,
        "seed": args["seed"],
        "speed": args["speed"],
        "temporal_agg": args["temporal_agg"],
        "camera_names": camera_names,
        "real_robot": not is_sim,
        "dataset_path": dataset_dir
    }

    if is_eval:
        ckpt_names = [f"policy_best.ckpt"]  # 10000 for insertion, 6500 for transfer
        results = []
        for ckpt_name in ckpt_names:
            success_rate, avg_return = eval_bc(config, ckpt_name, save_episode=True)
            results.append([ckpt_name, success_rate, avg_return])

        for ckpt_name, success_rate, avg_return in results:
            print(f"{ckpt_name}: {success_rate=} {avg_return=}")
        print()
        exit()
    
    if is_eval_speed:
        ckpt_names = [f"policy_last.ckpt"]  # 10000 for insertion, 6500 for transfer
        results = []
        for ckpt_name in ckpt_names:
            success_rate, avg_return = eval_speed_bc(config, ckpt_name, save_episode=True)
            results.append([ckpt_name, success_rate, avg_return])

        for ckpt_name, success_rate, avg_return in results:
            print(f"{ckpt_name}: {success_rate=} {avg_return=}")
        print()
        exit()
    
    if is_label:
        ckpt_names = [f"policy_last.ckpt"]  # 10000 for insertion, 6500 for transfer
        results = []
        for ckpt_name in ckpt_names:
            success_rate, avg_return = label_entropy(config, ckpt_name, save_episode=True)
            results.append([ckpt_name, success_rate, avg_return])
        exit()

    train_dataloader, val_dataloader, stats, _ = load_data(
        dataset_dir,
        num_episodes,
        camera_names,
        batch_size_train,
        batch_size_val,
        use_waypoint,
        constant_waypoint,
    )

    # save dataset stats
    if not os.path.isdir(ckpt_dir):
        os.makedirs(ckpt_dir)
    stats_path = os.path.join(ckpt_dir, f"dataset_stats.pkl")
    with open(stats_path, "wb") as f:
        pickle.dump(stats, f)

    best_ckpt_info = train_bc(train_dataloader, val_dataloader, config)
    best_epoch, min_val_loss, best_state_dict = best_ckpt_info

    # save best checkpoint
    ckpt_path = os.path.join(ckpt_dir, f"policy_best.ckpt")
    torch.save(best_state_dict, ckpt_path)
    print(f"Best ckpt, val loss {min_val_loss:.6f} @ epoch{best_epoch}")

KDE = KDE()
def speed_awe_entropy(actions, entropy, threshold):

        # waypoints = dp_waypoint_selection(actions=actions, err_threshold=0.005, pos_only=False)
        actions = actions.cpu().numpy()
        entropy = entropy.cpu().numpy()
        waypoints = dp_entropy_waypoint_selection(actions=actions, entropy=entropy, err_threshold=1, pos_only=False)
        actions = actions[waypoints]
        waypoints.insert(0,0)
        actions = torch.from_numpy(actions).cuda().unsqueeze(0)
        return actions

def eval_speed_bc(config, ckpt_name, save_episode=True):
    set_seed(100)
    ckpt_dir = config["ckpt_dir"]
    state_dim = config["state_dim"]
    real_robot = config["real_robot"]
    policy_class = config["policy_class"]
    onscreen_render = config["onscreen_render"]
    policy_config = config["policy_config"]
    camera_names = config["camera_names"]
    max_timesteps = config["episode_len"]
    task_name = config["task_name"]
    temporal_agg = config["temporal_agg"]
    speed = config["speed"]
    onscreen_cam = "angle"
   
    # load policy and stats
    ckpt_path = os.path.join(ckpt_dir, ckpt_name)
    policy = make_policy(policy_class, policy_config)
    loading_status = policy.load_state_dict(torch.load(ckpt_path))
    print(loading_status)
    policy.cuda()
    policy.eval()
    print(f"Loaded: {ckpt_path}")
    stats_path = os.path.join(ckpt_dir, f"dataset_stats.pkl")
    with open(stats_path, "rb") as f:
        stats = pickle.load(f)

    pre_process = lambda s_qpos: (s_qpos - stats["qpos_mean"]) / stats["qpos_std"]
    post_process = lambda a: a * stats["action_std"] + stats["action_mean"]
    # load environment
    if real_robot:
        from aloha_scripts.robot_utils import move_grippers  # requires aloha
        from aloha_scripts.real_env import make_real_env  # requires aloha

        env = make_real_env(init_node=True)
        env_max_reward = 0
    else:
        from act.sim_env import make_sim_env
        
        env = make_sim_env(task_name)
        env_max_reward = env.task.max_reward

    query_frequency = policy_config["num_queries"] 
    if temporal_agg:
        query_frequency = 1
        num_queries = policy_config["num_queries"]

    max_timesteps = int(max_timesteps * 1)  # may increase for real-world tasks
    max_timesteps = max_timesteps
    num_rollouts = 10
    episode_returns = []
    highest_rewards = []
    max_entropy_list = []
    min_entropy_list = []
    success_timesteps = 0
    for rollout_id in range(num_rollouts):
        rollout_id += 0
        ### set task
        if "sim_transfer_cube" in task_name:
            BOX_POSE[0] = sample_box_pose()  # used in sim reset
        elif "sim_insertion" in task_name:
            BOX_POSE[0] = np.concatenate(sample_insertion_pose())  # used in sim reset
        
        ts = env.reset()
 
        ### onscreen render
        if onscreen_render:
            ax = plt.subplot()
            plt_img = ax.imshow(
                env._physics.render(height=480, width=640, camera_id=onscreen_cam)
            )
            plt.ion()
        ### evaluation loop
        if temporal_agg:
            all_time_actions = torch.zeros(
                [max_timesteps, max_timesteps + num_queries, state_dim]
            ).cuda()
            all_time_entropy = torch.zeros(
                [max_timesteps, max_timesteps + num_queries, 1]
            ).cuda()
            all_time_samples = torch.zeros(
                [max_timesteps, max_timesteps + num_queries, 10,state_dim]
            ).cuda()
 
        qpos_history = torch.zeros((1, max_timesteps, state_dim)).cuda()
        image_list = []  # for visualization
        qpos_list = []
        target_qpos_list = []
        rewards = []
        traj_action_entropy = []
        flag = False
        waypoint_count = 0
        openloop_t = 0
        last_t = 0
        timestep_count = 0
        policy_slow = False
        real_speed = 1
        with torch.inference_mode():
            for t in tqdm(range(max_timesteps)):
                ### update onscreen render and wait for DT
                if onscreen_render:
                    image = env._physics.render(
                        height=480, width=640, camera_id=onscreen_cam
                    )
                    plt_img.set_data(image)
                    plt.pause(DT)

                ### process previous timestep to get qpos and image_list
                obs = ts.observation
                # if "images" in obs:
                #     image_list.append(obs["images"])
                # else:
                #     image_list.append({"main": obs["image"]})
                qpos_numpy = np.array(obs["qpos"])
                qpos = pre_process(qpos_numpy)
                qpos = torch.from_numpy(qpos).float().cuda().unsqueeze(0)
                qpos_history[:, t] = qpos
                curr_image = get_image(ts, camera_names)

                ### query policy
                if config["policy_class"] == "ACT" or config["policy_class"] == "DP":
                    if t % query_frequency == 0: # openloop_t == waypoint_count:
                        # get entropy
                        action_samples = policy.get_samples(qpos, curr_image)
                        all_actions = action_samples[[0]]
                        action_samples = action_samples.squeeze().permute(1,0,2) # (chunk_len, num_samples, dim)
                        entropy = torch.mean(torch.std(action_samples,dim=1),dim=-1)
                        # all_actions = all_actions[:,::speed]
                        # all_actions = speed_awe_entropy(all_actions.squeeze(), entropy, threshold=0.002)
                        waypoint_count = all_actions.shape[1]
                        openloop_t = 0
                        
                            
                    if temporal_agg:
                        if not policy_slow:
                            real_speed = speed
                            all_speed_actions = all_actions[:,::speed]
                            all_time_actions[[t], t:t + all_speed_actions.shape[1]] = all_speed_actions
                            all_time_samples[[t], t:t+ all_speed_actions.shape[1]] = action_samples[::speed]
                            
                            actions_for_curr_step = all_time_actions[:, t]
                            actions_populated = torch.all(
                                actions_for_curr_step != 0, axis=1
                            )
                            actions_for_curr_step = actions_for_curr_step[actions_populated]
                            actions_for_next_step = all_time_actions[:, t] # t+10
                            samples_populated = torch.all(
                                actions_for_next_step != 0, axis=1
                            )
                            samples_for_curr_step = all_time_samples[:, t]
                            samples_for_curr_step = samples_for_curr_step[samples_populated]
                            
                            entropy = torch.mean(torch.std(samples_for_curr_step.flatten(0,1),dim=0),dim=-1)
                            exp_weights = np.exp(-0.01 * np.arange(len(actions_for_curr_step)))
                            exp_weights = exp_weights / exp_weights.sum()
                            exp_weights = (
                                torch.from_numpy(exp_weights).cuda().unsqueeze(dim=1)
                            )
                            
                            # entropy = (math.log(torch.mean(entropy)+1e-8,1.5))
                            query_action = None
                            # For insertion 2x
                            # entropy = torch.tensor((entropy+37)/34.5).cuda()  # 20%：0.75
                            # For transfer 3x
                            # entropy = torch.tensor((entropy+37)/32).cuda() # 0.82
                            k = 0.01                            
                            # if t>20 : # and entropy <0.75:
                            #     actions_for_curr_step = minimizing_entropy_sampling(query_action.squeeze(),actions_for_curr_step.squeeze(),num_samples=13)

                           
                        if  False: # entropy >0.001 or policy_slow:
                            real_speed = 3
                            # slow policy
                            all_speed_actions = all_actions[:,::3]
                            all_time_actions[[t], t:t + all_speed_actions.shape[1]] = all_speed_actions
                            actions_for_curr_step = all_time_actions[:, t]
                            actions_populated = torch.all(
                                actions_for_curr_step != 0, axis=1
                            )
                            actions_for_curr_step = actions_for_curr_step[actions_populated]
                            
                            all_time_samples[[t], t:t+ all_speed_actions.shape[1]] = action_samples[::3]
                            
                            samples_for_curr_step = all_time_samples[:, t]
                            samples_for_curr_step = samples_for_curr_step[samples_populated]
                            exp_weights = np.exp(-0.01 * np.arange(len(actions_for_curr_step)))
                            exp_weights = exp_weights / exp_weights.sum()
                            exp_weights = (
                                torch.from_numpy(exp_weights).cuda().unsqueeze(dim=1)
                            )
                            query_action = (actions_for_curr_step * exp_weights).sum(
                            dim=0, keepdim=True
                            ) 
                            # _, anchor_action = KDE.kde_entropy(actions_for_curr_step.unsqueeze(0),k=1)
                            # actions_for_curr_step = minimizing_entropy_sampling(anchor_action.squeeze(),actions_for_curr_step.squeeze(),num_samples=min(actions_for_curr_step.shape[0],15))
                            
                            entropy = torch.mean(torch.std(samples_for_curr_step.flatten(0,1),dim=0),dim=-1)
                            exp_weights = np.exp(-0.01 * np.arange(len(actions_for_curr_step)))
                            exp_weights = (
                                torch.from_numpy(exp_weights).cuda().unsqueeze(dim=1)
                            )
                            
                            k = 0.01 
                            policy_slow = True
                            # Change policy_slow flag if entropy is large
                            if entropy>0.001:
                                policy_slow = False
                        
                        # k = 0.01 
                        exp_weights = np.exp(-k * np.arange(len(actions_for_curr_step)))
                        exp_weights = exp_weights / exp_weights.sum()
                        exp_weights = (
                            torch.from_numpy(exp_weights).cuda().unsqueeze(dim=1)
                        )
                        raw_action = (actions_for_curr_step * exp_weights).sum(
                            dim=0, keepdim=True
                        )
                        # if query_action is not None:
                        #     raw_action = 0.8*raw_action + 0.2*query_action
                        traj_action_entropy.append(entropy.squeeze())
                            
                    else:
                        # raw_action = all_actions[:, t % query_frequency]
                        raw_action = all_actions[:, openloop_t]
                        traj_action_entropy.append(entropy[t % query_frequency])
                        openloop_t += 1

                elif config["policy_class"] == "CNNMLP":
                    raw_action = policy(qpos, curr_image)
                else:
                    raise NotImplementedError
                
                # store processed image for video 
                entropy_numpy = np.array(traj_action_entropy[-1].cpu())
                store_imgs = {}
                for key, img in obs["images"].items():
                    store_imgs[key] = put_text(img,entropy_numpy)
                    store_imgs[key] = put_text(store_imgs[key],real_speed,position='bottom')
                if "images" in obs:
                    image_list.append(store_imgs)
                else:
                    image_list.append({"main": store_imgs})
                ### post-process actions
                raw_action = raw_action.squeeze(0).cpu().numpy()
                action = post_process(raw_action)
                target_qpos = action

                ### step the environment
                ### close-loop at gripper open/close destroy the performance
                ts = env.step(target_qpos)
                timestep_count += 1
                qpos_list.append(qpos_numpy)
                target_qpos_list.append(target_qpos)
                rewards.append(ts.reward)
                # compensate for imperfect gripper
                  
                    
                if np.array(ts.reward) == env_max_reward:
                    # timestep_count -= 1
                    # print(t)
                    success_timesteps += timestep_count
                    break

            plt.close()
        if real_robot:
            move_grippers(
                [env.puppet_bot_left, env.puppet_bot_right],
                [PUPPET_GRIPPER_JOINT_OPEN] * 2,
                move_time=0.5,
            )  # open
            pass

        rewards = np.array(rewards)
        episode_return = np.sum(rewards[rewards != None])
        episode_returns.append(episode_return)
        episode_highest_reward = np.max(rewards)
        highest_rewards.append(episode_highest_reward)
        print(
            f"Rollout {rollout_id}\n{episode_return=}, {episode_highest_reward=}, {env_max_reward=}, Success: {episode_highest_reward==env_max_reward}"
        )
        print(f"Total time count:{timestep_count}")
        traj_action_entropy = torch.stack(traj_action_entropy)
        traj_action_entropy = np.array(traj_action_entropy.cpu())
        print(traj_action_entropy.shape)
        print(f"max:{np.max(traj_action_entropy[:])} min:{np.min(traj_action_entropy[:])}")

        qpos = np.array(qpos_list)  # ts, dim
        from act.convert_ee import get_xyz

        left_arm_xyz = get_xyz(qpos[:, :6])
        right_arm_xyz = get_xyz(qpos[:, 7:13])
        # Find global min and max for each axis
        all_data = np.concatenate([left_arm_xyz, right_arm_xyz], axis=0)
        min_x, min_y, min_z = np.min(all_data, axis=0)
        max_x, max_y, max_z = np.max(all_data, axis=0)

        fig = plt.figure(figsize=(20, 10))
        ax1 = fig.add_subplot(121, projection="3d") 
        ax1.set_xlabel("x")
        ax1.set_ylabel("y")
        ax1.set_zlabel("z")
        ax1.set_title("Left", fontsize=20)
        ax1.set_xlim([min_x, max_x])
        ax1.set_ylim([min_y, max_y])
        ax1.set_zlim([min_z, max_z])
        from act.act_utils import plot_3d_trajectory
        plot_3d_trajectory(ax1, left_arm_xyz, traj_action_entropy,label="policy rollout", legend=False)

        ax2 = fig.add_subplot(122, projection="3d")
        ax2.set_xlabel("x")
        ax2.set_ylabel("y")
        ax2.set_zlabel("z")
        ax2.set_title("Right", fontsize=20)
        ax2.set_xlim([min_x, max_x])
        ax2.set_ylim([min_y, max_y])
        ax2.set_zlim([min_z, max_z])

        plot_3d_trajectory(ax2, right_arm_xyz, traj_action_entropy,label="policy rollout", legend=False)
        fig.suptitle(f"Task: {task_name}", fontsize=30) 

        handles, labels = ax1.get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=2, fontsize=20)
        
        if save_episode:
            save_videos(
                image_list,
                DT,
                video_path=os.path.join(ckpt_dir, f"video/video{rollout_id}-{speed}x3x.mp4"),
            )
            fig.savefig(
                os.path.join(ckpt_dir, f"plot/rollout{rollout_id}-{speed}x3x.png")
            )
            ax1.view_init(elev=90, azim=45)
            ax2.view_init(elev=90, azim=45)
            fig.savefig(
                os.path.join(ckpt_dir, f"plot/rollout{rollout_id}-{speed}x3x_view.png")
            )
        
        n_groups = qpos_numpy.shape[-1]
        tstep = np.linspace(0, 1, len(qpos_list)-1) 
        fig, axes = plt.subplots(nrows=n_groups, ncols=1, figsize=(8, 2 * n_groups), sharex=True)

        for n, ax in enumerate(axes):
            ax.plot(tstep, np.array(qpos_list)[1:, n], label=f'real qpos {n}')
            ax.plot(tstep, np.array(target_qpos_list)[:-1, n], label=f'target qpos {n}')
            ax.set_title(f'qpos {n}')
            ax.legend()
        
        plt.xlabel('timestep')
        plt.ylabel('qpos')
        plt.tight_layout()
        # fig.savefig(
        #         os.path.join(ckpt_dir, f"plot/rollout{rollout_id}_qpos.png")
        #     )
        plt.close()
        print(f"Save qpos curve to {ckpt_dir}/plot/rollout{rollout_id}_qpos.png")
        

    success_rate = np.mean(np.array(highest_rewards) == env_max_reward)
    avg_return = np.mean(episode_returns)
    summary_str = f"\nSuccess rate: {success_rate}\nAverage return: {avg_return}\nAverage consuming timesteps: {success_timesteps/(success_rate*len(episode_returns))}\n\n"
    for r in range(env_max_reward + 1):
        more_or_equal_r = (np.array(highest_rewards) >= r).sum()
        more_or_equal_r_rate = more_or_equal_r / num_rollouts
        summary_str += f"Reward >= {r}: {more_or_equal_r}/{num_rollouts} = {more_or_equal_r_rate*100}%\n"

    print(summary_str)

    # save success rate to txt
    result_file_name = "result_" + ckpt_name.split(".")[0] + ".txt"
    with open(os.path.join(ckpt_dir, result_file_name), "w") as f:
        f.write(summary_str)
        f.write(repr(episode_returns))
        f.write("\n\n")
        f.write(repr(highest_rewards))
    print(f"max entropy:{np.min(np.array(max_entropy_list))} min entropy:{np.max(np.array(min_entropy_list))}")
    return success_rate, avg_return

def label_entropy(config, ckpt_name, save_demos=False,save_episode=True):
    set_seed(2)
    ckpt_dir = config["ckpt_dir"]
    state_dim = config["state_dim"]
    real_robot = config["real_robot"]
    policy_class = config["policy_class"]
    onscreen_render = config["onscreen_render"]
    policy_config = config["policy_config"]
    camera_names = config["camera_names"]
    max_timesteps = config["episode_len"]
    task_name = config["task_name"]
    temporal_agg = config["temporal_agg"]
    onscreen_cam = "angle"
    variance_step = 1
    # load policy and stats
    ckpt_path = os.path.join(ckpt_dir, ckpt_name)
    policy = make_policy(policy_class, policy_config)
    loading_status = policy.load_state_dict(torch.load(ckpt_path))
    print(loading_status)
    policy.cuda()
    policy.eval()
    print(f"Loaded: {ckpt_path}")
    stats_path = os.path.join(ckpt_dir, f"dataset_stats.pkl")
    with open(stats_path, "rb") as f:
        stats = pickle.load(f)
    
    dataset_dir = config["dataset_path"]
    pre_process = lambda s_qpos: (s_qpos - stats["qpos_mean"]) / stats["qpos_std"]
    post_process = lambda a: a * stats["action_std"] + stats["action_mean"]
    # load environment
    if real_robot:
        from aloha_scripts.robot_utils import move_grippers  # requires aloha
        from aloha_scripts.real_env import make_real_env  # requires aloha

        env = make_real_env(init_node=True)
        env_max_reward = 0
    else:
        from act.sim_env import make_sim_env
        from act.act_utils import put_text, plot_3d_trajectory

        env = make_sim_env(task_name)
        env_max_reward = env.task.max_reward
    
    query_frequency = policy_config["num_queries"]
    if True: # temporal_agg: 
        query_frequency = 1
        num_queries = policy_config["num_queries"]

    max_timesteps = int(max_timesteps * 1)  # may increase for real-world tasks

    num_rollouts = 50
    episode_returns = []
    highest_rewards = []
    save_id = 0
    for rollout_id in range(num_rollouts):
        dataset_path = os.path.join(dataset_dir, f"episode_{rollout_id}.hdf5")
        with h5py.File(dataset_path, "r") as root:
            is_sim = root.attrs["sim"]
            original_action_shape = root["/action"].shape
            all_qpos = root["/observations/qpos"][()]
            image_dict = dict()
            for cam_name in camera_names:
                image_dict[cam_name] = root[f"/observations/images/{cam_name}"][()]
            all_cam_images = []
            for cam_name in camera_names:
                all_cam_images.append(image_dict[cam_name])
            all_cam_images = np.stack(all_cam_images, axis=1)
            print(all_cam_images.shape)
            print(all_qpos.shape)
        

        ### evaluation loop
        if True: # temporal_agg:
            all_time_actions = torch.zeros(
                [max_timesteps, max_timesteps + num_queries, state_dim]
            ).cuda()
            all_time_entropy = torch.zeros(
                [max_timesteps, max_timesteps + num_queries, 1]
            ).cuda()
            all_time_samples = torch.zeros(
                [max_timesteps, max_timesteps + num_queries, 10,state_dim]
            ).cuda()
            

        qpos_history = torch.zeros((1, max_timesteps, state_dim)).cuda()
        image_list = []  # for visualization
        qpos_list = []
        target_qpos_list = []
        rewards = []
        actions_var = []
        traj_action_entropy = []

        with torch.inference_mode():
            for t in tqdm(range(max_timesteps)):
                
                ### process previous timestep to get qpos and image_list
                qpos_numpy = all_qpos[t] 
                qpos = pre_process(qpos_numpy)
                qpos = torch.from_numpy(qpos).float().cuda().unsqueeze(0)
                qpos_history[:, t] = qpos
                curr_image = all_cam_images[t]
                obs = {}
                obs['images'] = {'top':curr_image[0]}
                curr_image = torch.from_numpy(curr_image).float().cuda().unsqueeze(0)                
                curr_image = curr_image.permute(0,1,4,2,3)
                
             
                ### query policy
                if config["policy_class"] == "ACT":
                    if t % query_frequency == 0:
                        action_samples = policy.get_samples(qpos, curr_image)
                        all_actions = action_samples[[0]]
                        
                    if True: # temporal_agg:
                        all_time_actions[[t], t : t + num_queries] = all_actions
                        all_time_samples[[t], t : t+ num_queries] = action_samples.permute(1,2,0,3)
                            
                        actions_for_curr_step = all_time_actions[:, t]
                        actions_populated = torch.all(
                            actions_for_curr_step != 0, axis=1
                        )
                        actions_for_curr_step = actions_for_curr_step[actions_populated]
                        actions_for_next_step = all_time_actions[:, t] # t+10
                        samples_populated = torch.all(
                            actions_for_next_step != 0, axis=1
                        )
                        samples_for_curr_step = all_time_samples[:, t]
                        samples_for_curr_step = samples_for_curr_step[samples_populated]
                        
                        entropy = torch.mean(torch.std(samples_for_curr_step.flatten(0,1),dim=0),dim=-1)
                        exp_weights = np.exp(-0.01 * np.arange(len(actions_for_curr_step)))
                        exp_weights = exp_weights / exp_weights.sum()
                        exp_weights = (
                            torch.from_numpy(exp_weights).cuda().unsqueeze(dim=1)
                        )
                            
                        raw_action = (actions_for_curr_step * exp_weights).sum(
                            dim=0, keepdim=True
                        )

                    traj_action_entropy.append(entropy)
                elif config["policy_class"] == "CNNMLP":
                    raw_action = policy(qpos, curr_image)
                else:
                    raise NotImplementedError
                    
                ### store processed image for video 
                entropy_numpy = np.array(traj_action_entropy[-1].cpu())
                store_imgs = {}
                for key, img in obs["images"].items():
                    store_imgs[key] = put_text(img,entropy_numpy)
                if "images" in obs:
                    image_list.append(store_imgs)
                else:
                    image_list.append({"main": store_imgs})
   
            plt.close()

        # draw trajectory curves
        traj_action_entropy = torch.stack(traj_action_entropy)
        traj_action_entropy = np.array(traj_action_entropy.cpu())

        actions_entropy_norm = traj_action_entropy

        qpos = all_qpos  # ts, dim
        from act.convert_ee import get_xyz

        left_arm_xyz = get_xyz(qpos[:, :6])
        right_arm_xyz = get_xyz(qpos[:, 7:13])
        # Find global min and max for each axis
        all_data = np.concatenate([left_arm_xyz, right_arm_xyz], axis=0)
        min_x, min_y, min_z = np.min(all_data, axis=0)
        max_x, max_y, max_z = np.max(all_data, axis=0)

        fig = plt.figure(figsize=(20, 10))
        ax1 = fig.add_subplot(121, projection="3d") 
        ax1.set_xlabel("x")
        ax1.set_ylabel("y")
        ax1.set_zlabel("z")
        ax1.set_title("Left", fontsize=20)
        ax1.set_xlim([min_x, max_x])
        ax1.set_ylim([min_y, max_y])
        ax1.set_zlim([min_z, max_z])
        
        plot_3d_trajectory(ax1, left_arm_xyz, actions_entropy_norm,label="policy rollout", legend=False)

        ax2 = fig.add_subplot(122, projection="3d")
        ax2.set_xlabel("x")
        ax2.set_ylabel("y")
        ax2.set_zlabel("z")
        ax2.set_title("Right", fontsize=20)
        ax2.set_xlim([min_x, max_x])
        ax2.set_ylim([min_y, max_y])
        ax2.set_zlim([min_z, max_z])

        plot_3d_trajectory(ax2, right_arm_xyz, actions_entropy_norm,label="policy rollout", legend=False)
        fig.suptitle(f"Task: {task_name}", fontsize=30) 

        handles, labels = ax1.get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=2, fontsize=20)
                
        # Only save successful video/plot
        if save_episode : #and episode_highest_reward==env_max_reward: 
            save_videos(
                image_list,
                DT,
                video_path=os.path.join(ckpt_dir, f"video/rollout{rollout_id}.mp4"),
            )
            fig.savefig(
                os.path.join(ckpt_dir, f"plot/rollout{rollout_id}.png")
            )
            ax1.view_init(elev=90, azim=45)
            ax2.view_init(elev=90, azim=45)
            fig.savefig(
                os.path.join(ckpt_dir, f"plot/rollout{rollout_id}_demos{save_id}_view.png")
            )
            
        plt.close(fig)
        save_labels = True
        if save_labels:
            with h5py.File(dataset_path, "r+") as root:
                name = f"/entropy"
                try:
                    root[name] = actions_entropy_norm
                except:
                    del root[name]
                    root[name] = actions_entropy_norm  

    
    return success_rate, avg_return
    

def make_policy(policy_class, policy_config):
    if policy_class == "ACT":
        policy = ACTPolicy(policy_config)
    elif policy_class == "CNNMLP":
        policy = CNNMLPPolicy(policy_config)
    elif policy_class == "DP":
        policy = DiffusionPolicy(policy_config)
    else:
        raise NotImplementedError
    return policy


def make_optimizer(policy_class, policy):
    if policy_class == "ACT":
        optimizer = policy.configure_optimizers()
    elif policy_class == "CNNMLP":
        optimizer = policy.configure_optimizers()
    elif policy_class == "DP":
        optimizer = policy.configure_optimizers()
    else:
        raise NotImplementedError
    return optimizer


def get_image(ts, camera_names):
    curr_images = []
    for cam_name in camera_names:
        curr_image = rearrange(ts.observation["images"][cam_name], "h w c -> c h w")
        curr_images.append(curr_image)
    curr_image = np.stack(curr_images, axis=0)
    curr_image = torch.from_numpy(curr_image / 255.0).float().cuda().unsqueeze(0)
    return curr_image


def eval_bc(config, ckpt_name, save_episode=True):
    set_seed(200)
    ckpt_dir = config["ckpt_dir"]
    state_dim = config["state_dim"]
    real_robot = config["real_robot"]
    policy_class = config["policy_class"]
    onscreen_render = config["onscreen_render"]
    policy_config = config["policy_config"]
    camera_names = config["camera_names"]
    max_timesteps = config["episode_len"]
    task_name = config["task_name"]
    temporal_agg = config["temporal_agg"]
    onscreen_cam = "angle"

    # load policy and stats
    ckpt_path = os.path.join(ckpt_dir, ckpt_name)
    policy = make_policy(policy_class, policy_config)
    loading_status = policy.load_state_dict(torch.load(ckpt_path))
    print(loading_status)
    policy.cuda()
    policy.eval()
    print(f"Loaded: {ckpt_path}")
    stats_path = os.path.join(ckpt_dir, f"dataset_stats.pkl")
    with open(stats_path, "rb") as f:
        stats = pickle.load(f)

    pre_process = lambda s_qpos: (s_qpos - stats["qpos_mean"]) / stats["qpos_std"]
    post_process = lambda a: a * stats["action_std"] + stats["action_mean"]

    # load environment
    if real_robot:
        from aloha_scripts.robot_utils import move_grippers  # requires aloha
        from aloha_scripts.real_env import make_real_env  # requires aloha

        env = make_real_env(init_node=True)
        env_max_reward = 0
    else:
        from act.sim_env import make_sim_env

        env = make_sim_env(task_name)
        env_max_reward = env.task.max_reward

    query_frequency = policy_config["num_queries"]
    if temporal_agg:
        query_frequency = 1
        num_queries = policy_config["num_queries"]

    max_timesteps = int(max_timesteps * 1)  # may increase for real-world tasks

    num_rollouts = 50
    episode_returns = []
    highest_rewards = []
    for rollout_id in range(num_rollouts):
        rollout_id += 0
        ### set task
        if "sim_transfer_cube" in task_name:
            BOX_POSE[0] = sample_box_pose()  # used in sim reset
        elif "sim_insertion" in task_name:
            BOX_POSE[0] = np.concatenate(sample_insertion_pose())  # used in sim reset

        ts = env.reset()

        ### onscreen render
        if onscreen_render:
            ax = plt.subplot()
            plt_img = ax.imshow(
                env._physics.render(height=480, width=640, camera_id=onscreen_cam)
            )
            plt.ion()

        ### evaluation loop
        if temporal_agg:
            all_time_actions = torch.zeros(
                [max_timesteps, max_timesteps + num_queries, state_dim]
            ).cuda()

        qpos_history = torch.zeros((1, max_timesteps, state_dim)).cuda()
        image_list = []  # for visualization
        qpos_list = []
        target_qpos_list = []
        rewards = []
        with torch.inference_mode():
            for t in tqdm(range(max_timesteps)):
                ### update onscreen render and wait for DT
                if onscreen_render:
                    image = env._physics.render(
                        height=480, width=640, camera_id=onscreen_cam
                    )
                    plt_img.set_data(image)
                    plt.pause(DT)

                ### process previous timestep to get qpos and image_list
                obs = ts.observation
                if "images" in obs:
                    image_list.append(obs["images"])
                else:
                    image_list.append({"main": obs["image"]})
                qpos_numpy = np.array(obs["qpos"])
                qpos = pre_process(qpos_numpy)
                qpos = torch.from_numpy(qpos).float().cuda().unsqueeze(0)
                qpos_history[:, t] = qpos
                curr_image = get_image(ts, camera_names)
                
                ### query policy
                query_frequency = 48
                if config["policy_class"] == "ACT" or config["policy_class"] == "DP":
                    if t % query_frequency == 0:
                        all_actions = policy(qpos, curr_image)
                    if temporal_agg:
                        all_time_actions[[t], t : t + num_queries] = all_actions
                        actions_for_curr_step = all_time_actions[:, t]
                        actions_populated = torch.all(
                            actions_for_curr_step != 0, axis=1
                        )
                        actions_for_curr_step = actions_for_curr_step[actions_populated]
                        k = 0.01
                        exp_weights = np.exp(-k * np.arange(len(actions_for_curr_step)))
                        exp_weights = exp_weights / exp_weights.sum()
                        exp_weights = (
                            torch.from_numpy(exp_weights).cuda().unsqueeze(dim=1)
                        )
                        raw_action = (actions_for_curr_step * exp_weights).sum(
                            dim=0, keepdim=True
                        )
                    else:
                        raw_action = all_actions[:, t % query_frequency]
                elif config["policy_class"] == "CNNMLP":
                    raw_action = policy(qpos, curr_image)
                else:
                    raise NotImplementedError

                ### post-process actions
                raw_action = raw_action.squeeze(0).cpu().numpy()
                action = post_process(raw_action)
                target_qpos = action

                ### step the environment
                ts = env.step(target_qpos)

                ### for visualization
                qpos_list.append(qpos_numpy)
                target_qpos_list.append(target_qpos)
                rewards.append(ts.reward)

                if np.array(ts.reward) == env_max_reward:
                    break

            plt.close()
        if real_robot:
            move_grippers(
                [env.puppet_bot_left, env.puppet_bot_right],
                [PUPPET_GRIPPER_JOINT_OPEN] * 2,
                move_time=0.5,
            )  # open
            pass

        rewards = np.array(rewards)
        episode_return = np.sum(rewards[rewards != None])
        episode_returns.append(episode_return)
        episode_highest_reward = np.max(rewards)
        highest_rewards.append(episode_highest_reward)
        print(
            f"Rollout {rollout_id}\n{episode_return=}, {episode_highest_reward=}, {env_max_reward=}, Success: {episode_highest_reward==env_max_reward}"
        )

        if save_episode:
            save_videos(
                image_list,
                DT,
                video_path=os.path.join(ckpt_dir, f"video{rollout_id}x2.mp4"),
            )
        
        n_groups = qpos_numpy.shape[-1]
        tstep = np.linspace(0, 1, len(qpos_list)-1) 
        fig, axes = plt.subplots(nrows=n_groups, ncols=1, figsize=(8, 2 * n_groups), sharex=True)

        for n, ax in enumerate(axes):
            ax.plot(tstep, np.array(qpos_list)[1:, n], label=f'real qpos {n}')
            ax.plot(tstep, np.array(target_qpos_list)[:-1, n], label=f'target qpos {n}')
            ax.set_title(f'qpos {n}')
            ax.legend()
        
        plt.xlabel('timestep')
        plt.ylabel('qpos')
        plt.tight_layout()
        fig.savefig(
               os.path.join(ckpt_dir, f"plot/rollout{rollout_id}_qposx2.png")
            )
        plt.close()
        print(f"Save qpos curve to {ckpt_dir}/plot/rollout{rollout_id}_qpos.png")

    success_rate = np.mean(np.array(highest_rewards) == env_max_reward)
    avg_return = np.mean(episode_returns)
    summary_str = f"\nSuccess rate: {success_rate}\nAverage return: {avg_return}\n\n"

    for r in range(env_max_reward + 1):
        more_or_equal_r = (np.array(highest_rewards) >= r).sum()
        more_or_equal_r_rate = more_or_equal_r / num_rollouts
        summary_str += f"Reward >= {r}: {more_or_equal_r}/{num_rollouts} = {more_or_equal_r_rate*100}%\n"

    print(summary_str)

    # save success rate to txt
    result_file_name = "result_" + ckpt_name.split(".")[0] + ".txt"
    with open(os.path.join(ckpt_dir, result_file_name), "w") as f:
        f.write(summary_str)
        f.write(repr(episode_returns))
        f.write("\n\n")
        f.write(repr(highest_rewards))

    return success_rate, avg_return


def forward_pass(data, policy):
    image_data, qpos_data, action_data, is_pad = data
    image_data, qpos_data, action_data, is_pad = (
        image_data.cuda(),
        qpos_data.cuda(),
        action_data.cuda(),
        is_pad.cuda(),
    )
    return policy(qpos_data, image_data, action_data, is_pad)  # TODO remove None


def train_bc(train_dataloader, val_dataloader, config):
    num_epochs = config["num_epochs"]
    ckpt_dir = config["ckpt_dir"]
    seed = config["seed"]
    policy_class = config["policy_class"]
    policy_config = config["policy_config"]

    set_seed(seed)

    policy = make_policy(policy_class, policy_config)
    # if ckpt_dir is not empty, prompt the user to load the checkpoint
    if os.path.isdir(ckpt_dir) and len(os.listdir(ckpt_dir)) > 1:
        print(f"Checkpoint directory {ckpt_dir} is not empty. Load checkpoint? (y/n)")
        load_ckpt = input()
        if load_ckpt == "y":
            # load the latest checkpoint
            latest_idx = max(
                [
                    int(f.split("_")[2])
                    for f in os.listdir(ckpt_dir)
                    if f.startswith("policy_epoch_")
                ]
            )
            ckpt_path = os.path.join(
                ckpt_dir, f"policy_epoch_{latest_idx}_seed_{seed}.ckpt"
            )
            print(f"Loading checkpoint from {ckpt_path}")
            loading_status = policy.load_state_dict(torch.load(ckpt_path))
            print(loading_status)
        else:
            print("Not loading checkpoint")
            latest_idx = 0
    else:
        latest_idx = 0

    policy.cuda()
    optimizer = make_optimizer(policy_class, policy)

    train_history = []
    validation_history = []
    min_val_loss = np.inf
    best_ckpt_info = None
    for epoch in tqdm(range(latest_idx, num_epochs)):
        print(f"\nEpoch {epoch}")
        # validation
        with torch.inference_mode():
            policy.eval()
            epoch_dicts = []
            for batch_idx, data in enumerate(val_dataloader):
                forward_dict = forward_pass(data, policy)
                epoch_dicts.append(forward_dict)
            epoch_summary = compute_dict_mean(epoch_dicts)
            validation_history.append(epoch_summary)

            epoch_val_loss = epoch_summary["loss"]
            if epoch_val_loss < min_val_loss:
                min_val_loss = epoch_val_loss
                best_ckpt_info = (epoch, min_val_loss, deepcopy(policy.state_dict()))
        print(f"Val loss:   {epoch_val_loss:.5f}")
        summary_string = ""
        for k, v in epoch_summary.items():
            summary_string += f"{k}: {v.item():.3f} "
        print(summary_string)

        # training
        policy.train()
        optimizer.zero_grad()
        for batch_idx, data in enumerate(train_dataloader):
            forward_dict = forward_pass(data, policy)
            # backward
            loss = forward_dict["loss"]
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            train_history.append(detach_dict(forward_dict))
        e = epoch - latest_idx
        epoch_summary = compute_dict_mean(
            train_history[(batch_idx + 1) * e : (batch_idx + 1) * (epoch + 1)]
        )
        epoch_train_loss = epoch_summary["loss"]
        print(f"Train loss: {epoch_train_loss:.5f}")
        summary_string = ""
        for k, v in epoch_summary.items():
            summary_string += f"{k}: {v.item():.3f} "
        print(summary_string)

        if epoch % 100 == 0:
            ckpt_path = os.path.join(ckpt_dir, f"policy_epoch_{epoch}_seed_{seed}.ckpt")
            torch.save(policy.state_dict(), ckpt_path)
            plot_history(train_history, validation_history, epoch, ckpt_dir, seed)

    ckpt_path = os.path.join(ckpt_dir, f"policy_last.ckpt")
    torch.save(policy.state_dict(), ckpt_path)

    best_epoch, min_val_loss, best_state_dict = best_ckpt_info
    ckpt_path = os.path.join(ckpt_dir, f"policy_epoch_{best_epoch}_seed_{seed}.ckpt")
    torch.save(best_state_dict, ckpt_path)
    print(
        f"Training finished:\nSeed {seed}, val loss {min_val_loss:.6f} at epoch {best_epoch}"
    )

    # save training curves
    plot_history(train_history, validation_history, num_epochs, ckpt_dir, seed)

    return best_ckpt_info


def plot_history(train_history, validation_history, num_epochs, ckpt_dir, seed):
    # save training curves
    for key in train_history[0]:
        plot_path = os.path.join(ckpt_dir, f"train_val_{key}_seed_{seed}.png")
        plt.figure()
        train_values = [summary[key].item() for summary in train_history]
        val_values = [summary[key].item() for summary in validation_history]
        plt.plot(
            np.linspace(0, num_epochs - 1, len(train_history)),
            train_values,
            label="train",
        )
        plt.plot(
            np.linspace(0, num_epochs - 1, len(validation_history)),
            val_values,
            label="validation",
        )
        # plt.ylim([-0.1, 1])
        plt.tight_layout()
        plt.legend()
        plt.title(key)
        plt.savefig(plot_path)
    print(f"Saved plots to {ckpt_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--label", action="store_true")
    parser.add_argument("--eval_speed", action="store_true")
    parser.add_argument("--onscreen_render", action="store_true")
    parser.add_argument(
        "--ckpt_dir", action="store", type=str, help="ckpt_dir", required=True
    )
    parser.add_argument(
        "--policy_class",
        action="store",
        type=str,
        help="policy_class, capitalize",
        required=True,
    )
    parser.add_argument(
        "--task_name", action="store", type=str, help="task_name", required=True
    )
    parser.add_argument(
        "--diffusion_policy_cfg", action="store", type=str, help="task_name", default='act/image_aloha_diffusion_policy_cnn.yaml'
    )
    parser.add_argument(
        "--batch_size", action="store", type=int, help="batch_size", required=True
    )
    parser.add_argument("--seed", action="store", type=int, help="seed", required=True)
    parser.add_argument("--speed", action="store", type=int, help="seed", default=1)
    parser.add_argument(
        "--num_epochs", action="store", type=int, help="num_epochs", required=True
    )
    parser.add_argument("--lr", action="store", type=float, help="lr", required=True)

    # for ACT
    parser.add_argument(
        "--kl_weight", action="store", type=int, help="KL Weight", required=False
    )
    parser.add_argument(
        "--chunk_size", action="store", type=int, help="chunk_size", required=False
    )
    parser.add_argument(
        "--hidden_dim", action="store", type=int, help="hidden_dim", required=False
    )
    parser.add_argument(
        "--dim_feedforward",
        action="store",
        type=int,
        help="dim_feedforward",
        required=False,
    )
    parser.add_argument("--temporal_agg", action="store_true")

    # for waypoints
    parser.add_argument("--use_waypoint", action="store_true")
    parser.add_argument(
        "--constant_waypoint",
        action="store",
        type=int,
        help="constant_waypoint",
        required=False,
    )

    main(vars(parser.parse_args()))
