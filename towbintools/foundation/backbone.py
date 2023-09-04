import networkx as nx
import numpy as np
import skan
from scipy import interpolate
from scipy.spatial import distance
from skimage.morphology import skeletonize


def skeletonize_and_skan(
    mask: np.ndarray,
):
    """Skeletonize a binary mask and convert the backbone to a skan Skeleton."""

    backbone = skeletonize(mask)
    skeleton = skan.Skeleton(backbone)
    return skeleton


def backbone_to_skan(
    backbone: np.ndarray,
):
    """Convert a backbone to a skan Skeleton."""

    return skan.Skeleton(backbone)


def backbone_to_longest_shortest_path(
    backbone: np.ndarray,
):
    """Return the coordinates of the longest shortest path of a given backbone."""

    skeleton = backbone_to_skan(backbone)
    main_branch = find_main_branch_paths(skeleton)
    longest_shortest_path_coordinates = paths_to_coordinates(skeleton, main_branch)
    return longest_shortest_path_coordinates


def skeleton_to_longest_shortest_path(
    skeleton: skan.Skeleton,
):
    """Return the coordinates of the longest shortest path of a given skeleton."""

    main_branch = find_main_branch_paths(skeleton)
    longest_shortest_path_coordinates = paths_to_coordinates(skeleton, main_branch)
    return longest_shortest_path_coordinates


def find_main_branch_paths(
    skeleton: skan.Skeleton,
):
    branch_data = skan.summarize(skeleton, find_main_branch=True)

    # get the largest subskeleton
    subskeletons = branch_data.groupby(by="skeleton-id", as_index=True)
    largest_subskeleton = subskeletons.get_group(
        subskeletons["branch-distance"].sum().idxmax()
    )

    # get the nodes/edges for skeleton's branch-graph
    main_branch_edges = largest_subskeleton.loc[
        largest_subskeleton.main == True, ["node-id-src", "node-id-dst"]
    ].to_numpy()
    main_branch_nodes, counts = np.unique(
        main_branch_edges.flatten(), return_counts=True
    )
    main_branch_graph = nx.Graph()
    main_branch_graph.add_edges_from(main_branch_edges)
    end1, end2 = main_branch_nodes[np.nonzero(counts == 1)]

    # order the nodes
    main_branch_nodes = np.asarray(nx.shortest_path(main_branch_graph, end1, end2))
    PATH_AXIS = 0
    NODE_AXIS = 1
    ALL_PATHS = np.s_[:]

    # main_branch_mask has True if path is part of main branch
    # path is part of main branch if both of its nodes are in main_branch_nodes, hence check for sum(NODE_AXIS) == 2
    main_branch_mask = (
        skeleton.paths[ALL_PATHS, main_branch_nodes].sum(axis=NODE_AXIS) == 2
    )
    main_branch_paths = np.flatnonzero(main_branch_mask)
    # subset skeleton.paths with unordered main_branch_paths and the ordered main_branch_nodes to give something like the following:
    # 0 1 1 0 0
    # 1 1 0 0 0
    # 0 0 1 1 0
    # 0 0 0 1 1
    # argmax for each row will give the col-index for the first "1" in that row
    # argsorting the col-index gives the path order
    # i.e. read the 1s in matrix from left to right, recording the order of the rows
    # for the example above, the desired result is [1, 0, 2, 3], which is the row order that would reorder the matrix to:
    # 1 1 0 0 0
    # 0 1 1 0 0
    # 0 0 1 1 0
    # 0 0 0 1 1
    path_order = (
        skeleton.paths[np.ix_(main_branch_paths, main_branch_nodes)]
        .argmax(axis=NODE_AXIS)
        .argsort(axis=PATH_AXIS)
        .A1  # A1 converts np.matrix to 1D np.array
    )
    main_branch_paths = main_branch_paths[path_order]

    return main_branch_paths


def paths_to_coordinates(
    skeleton: skan.Skeleton,
    paths,
):
    # axis 0 -> points; axis 1 -> xy coords
    POINTS = 0
    path_coords = [skeleton.path_coordinates(path) for path in paths]

    ordered_path_coords = []
    for path1_coords, path2_coords in zip(path_coords[:-1], path_coords[1:]):
        path1_end = path1_coords[-1]
        path2_tip1, path2_tip2 = path2_coords[[0, -1]]
        # path1_end is indeed the end
        if np.all(path1_end == path2_tip1) or np.all(path1_end == path2_tip2):
            ordered_path_coords.append(path1_coords)
        # path1_end is actually the start, so flip
        else:
            flipped = np.flip(path1_coords, axis=POINTS)
            ordered_path_coords.append(flipped)

    last_path = path_coords[-1]
    last_path_start = last_path[0]
    if len(ordered_path_coords) == 0:
        ordered_path_coords.append(last_path)
    else:
        last_ordered_coord = ordered_path_coords[-1][-1]
        if np.all(last_ordered_coord == last_path_start):
            # if end and start align, append as is
            ordered_path_coords.append(last_path)
        else:
            # if end and start don't align, flip first then append
            ordered_path_coords.append(np.flip(last_path, axis=POINTS))

    ordered_path_coords = np.concatenate(ordered_path_coords, axis=POINTS)
    # coords need to be unique because splprep will throw an error otherwise
    _, uniq_indices = np.unique(ordered_path_coords, return_index=True, axis=POINTS)
    uniq_indices.sort()
    ordered_path_coords = ordered_path_coords[uniq_indices]
    return ordered_path_coords


def generate_parametrisation(
    backbone,
    max_length,
):
    backbone_inter_pixel_dist = np.linalg.norm(backbone[:-1] - backbone[1:], axis=1)
    length = backbone_inter_pixel_dist.sum()
    u_per_pixel = 1 / max_length

    backbone_inter_u_dist = backbone_inter_pixel_dist * u_per_pixel
    start = (1 - length * u_per_pixel) / 2
    us = np.zeros(len(backbone))
    us[0] = start
    us[1:] = np.cumsum(backbone_inter_u_dist) + start
    return us


def calculate_worm_length(
    backbone,
):
    return np.linalg.norm(backbone[:-1] - backbone[1:], axis=1).sum()


def spline_from_backbone_coordinates(
    backbone,
    upsampling: int,
    smoothing=75,
):
    backbone_lengths = np.asarray(calculate_worm_length(backbone))
    max_length = backbone_lengths.max()
    ys, xs = backbone.T
    us = generate_parametrisation(backbone, max_length)
    tck, u = interpolate.splprep([ys, xs], u=us, ub=us[0], ue=us[-1], s=smoothing)
    t, c, k = tck
    c = np.asarray(c).T
    spline = interpolate.BSpline(t, c, k, extrapolate=False)
    return_us = np.linspace(us.min(), us.max(), len(us) * upsampling)
    return spline(return_us)


def are_tips_flipped(
    tip_pair1,
    tip_pair2,
):
    pair_wise_dists = distance.cdist(tip_pair1, tip_pair2, metric="euclidean")
    unflipped_dist = np.sum(pair_wise_dists[[0, 1], [0, 1]])
    flipped_dist = np.sum(pair_wise_dists[[0, 1], [1, 0]])
    return flipped_dist < unflipped_dist
