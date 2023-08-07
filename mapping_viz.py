import os
import glob
import platform
import argparse
import numpy as np
import matplotlib.pyplot as plt

from os.path import join
from HAFrame.distance import load_distance_matrix
from MBM.better_mistakes.util.config import load_config
from HAFrame.solve_HAF import map_hdistance_to_cosine_similarity_linear
from HAFrame.solve_HAF import find_max_separation_matrix_factorization_solver
from HAFrame.solve_HAF import map_hdistance_to_cosine_similarity_exponential_decay


DATASET_NAMES = [
    "tiered-imagenet-224",
    "inaturalist19-224",
    "cifar-100",
    "fgvc-aircraft"
]

parser = argparse.ArgumentParser()
parser.add_argument("--data", type=str, help="dataset name", choices=DATASET_NAMES)
parser.add_argument("--data_dir", default="data/", help="Folder containing the supplementary data")
parser.add_argument("--data-paths-config", help="Path to data paths yaml file", default="data_paths.yml")
viz_opts = parser.parse_args()


def get_sub_dir_list(home_dir):

    my_platform = platform.system()
    sub_dir_list = []
    if my_platform == 'Windows':
        split_char = '\\'
    else:
        split_char = '/'

    for sub_dir in glob.glob(join(home_dir, '*')):
        sub_dir_name = sub_dir.split(split_char)
        sub_dir_list.append(sub_dir_name[-1])
    return sub_dir_list


def viz_mapping(hdistance, mappings, labels, gammas, cosine_min, fig_path=None):
    fig = plt.figure(figsize=(12, 10))
    for k, map_func in enumerate(mappings):
        s = map_func(hdistance=hdistance, min_similarity=cosine_min[k], gamma=gammas[k])
        plt.plot(hdistance, s, '*-', label=labels[k])

    # plt.plot(hdistance, np.zeros_like(hdistance),"--", label='zero')
    plt.xlabel('hierarchical distance', fontsize=25)
    plt.ylabel('cosine similarity', fontsize=25)
    plt.ylim(-0.2, 1.1)
    plt.xticks(fontsize=25)
    plt.yticks(fontsize=25)
    plt.legend(fontsize='25')
    # plt.title("mapping from hdist to cosine similarity")
    plt.grid(True)
    if fig_path:
        fig.savefig(fig_path)
        print(f"saving figure at {fig_path}")

    # plt.show()
    return


def main(opts):

    if opts.data == "cifar-100":
        opts.data_dir = os.path.join(opts.data_dir, "cifar-l5/original/")

    data_paths = load_config(opts.data_paths_config)
    opts.data_path = data_paths[opts.data]
    input_dir = os.path.join(opts.data_path, "train")

    classes = sorted(get_sub_dir_list(input_dir))
    num_classes=len(classes)
    if opts.data == "cifar-100":
        classes = ["L5-" + str(i) for i in range(num_classes)]

    dist_filename_dict = {
        "fgvc-aircraft": "fgvc_aircraft_original_distances.pkl.xz",
        "cifar-100": "cifar_100_ilsvrc_distances.pkl.xz",
        "inaturalist19-224": "inaturalist19_ilsvrc_distances.pkl.xz",
        "tiered-imagenet-224": "tiered_imagenet_ilsvrc_distances.pkl.xz"
    }

    distance_path = os.path.join(opts.data_dir, dist_filename_dict[opts.data])

    # load distance as matrix
    hdist = load_distance_matrix(distance_path, classes)
    hdrange = np.arange(0, np.max(hdist) + 1)

    n_samples = 100
    # in Barz & Denzler, the pair-wise similarity is constraint to be >=0
    ret_min_sim_linear = 0
    """
    ret_min_sim_linear = find_max_separation_matrix_factorization_solver(
        hdist,
        num_classes,
        n_samples,
        map_hdistance_to_cosine_similarity_linear,
        None
    )
    """

    mappings = [
        map_hdistance_to_cosine_similarity_linear,
        map_hdistance_to_cosine_similarity_exponential_decay,  # 1
        map_hdistance_to_cosine_similarity_exponential_decay,  # 2
        map_hdistance_to_cosine_similarity_exponential_decay,  # 3
        map_hdistance_to_cosine_similarity_exponential_decay,  # 4
        map_hdistance_to_cosine_similarity_exponential_decay,  # 5
    ]

    hdist_max = np.max(hdist)
    gammas = [
        None,
        1, 2, 3, 4, 5,
    ]

    min_cos_sims = [ret_min_sim_linear] + [
        find_max_separation_matrix_factorization_solver(
            hdist,
            num_classes,
            n_samples,
            map_hdistance_to_cosine_similarity_exponential_decay,
            g
        ) for g in gammas[1:]
    ]

    labels = ['linear'] + [r"$\gamma$ = " + f"{g:.1f}" for g in gammas[1:]]
    fig_path = os.path.join(opts.data_dir, f"{opts.data}-mapping.jpg")
    viz_mapping(hdrange, mappings, labels, gammas, min_cos_sims, fig_path)


if __name__ == "__main__":
    main(viz_opts)