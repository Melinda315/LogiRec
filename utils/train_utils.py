import argparse
import networkx as nx
import torch
import numpy as np


def find_exclusion(user_tag_matrix, exclusion):

    exclusion = exclusion.long().cpu().numpy()
    num_user, num_tag = user_tag_matrix.shape
    tag_exclusion_matrix = np.zeros((num_tag, num_tag))
    for ex in exclusion:
        left = ex[0]
        right = ex[1]
        tag_exclusion_matrix[left][right] = 1

    user_tags = []
    user_tag_matrix = user_tag_matrix.numpy()
    for user_idx in range(num_user):
        tags = [tag_idx for tag_idx in range(num_tag) if user_tag_matrix[user_idx][tag_idx] == 1]
        user_tags.append(tags)

    user_exclusion_tag = []
    for user_tag in user_tags:
        tmp = []
        for tag1 in user_tag:
            for tag2 in user_tag:
                if tag1 == tag2: continue
                if tag_exclusion_matrix[tag1][tag2] == 1: tmp.append([tag1, tag2])
        user_exclusion_tag.append(tmp)
    return user_exclusion_tag


def generate_user_tag_matrix(user_item_matrix, item_tag_matrix):

    num_user, num_item = user_item_matrix.shape
    num_item, num_tag = item_tag_matrix.shape

    user_tag_matrix = np.zeros((num_user, num_tag))
    user_item_matrix = user_item_matrix.numpy()
    item_tag_matrix = item_tag_matrix.numpy()

    for user in range(num_user):
        user_items = user_item_matrix[user, :]
        item_indices = np.nonzero(user_items)[0]
        user_tags = np.sum(item_tag_matrix[item_indices, :], axis=0)
        user_tags[user_tags > 1] = 1
        user_tag_matrix[user, :] = user_tags

    return user_tag_matrix


def build_tree(parent_child_pairs):
    """
    Construct tag taxonomy and find the levels of tags
    """

    parent_child_pairs = parent_child_pairs.long().tolist()
    G = nx.DiGraph()

    for child, parent in parent_child_pairs:
        G.add_edge(parent, child)

    levels = {}
    for node in nx.topological_sort(G):
        parent = list(G.predecessors(node))
        if parent:
            levels[node] = levels[parent[0]] + 1
        else:
            levels[node] = 0

    ans = max(levels, key=lambda x: levels[x])
    max_level = levels[ans]

    return max_level, levels


def add_flags_from_config(parser, config_dict):
    """
    Adds a flag (and default value) to an ArgumentParser for each parameter in a config
    """

    def OrNone(default):
        def func(x):
            # Convert "none" to proper None object
            if x.lower() == "none":
                return None
            # If default is None (and x is not None), return x without conversion as str
            elif default is None:
                return str(x)
            # Otherwise, default has non-None type; convert x to that type
            else:
                return type(default)(x)

        return func

    for param in config_dict:
        default, description = config_dict[param]
        try:
            if isinstance(default, dict):
                parser = add_flags_from_config(parser, default)
            elif isinstance(default, list):
                if len(default) > 0:
                    # pass a list as argument
                    parser.add_argument(
                            f"--{param}",
                            action="append",
                            type=type(default[0]),
                            default=default,
                            help=description
                    )
                else:
                    pass
                    parser.add_argument(f"--{param}", action="append", default=default, help=description)
            else:
                pass
                parser.add_argument(f"--{param}", type=OrNone(default), default=default, help=description)
        except argparse.ArgumentError:
            print(
                f"Could not add flag for param {param} because it was already present."
            )
    return parser


