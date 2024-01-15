import sys

sys.path.append("..")

import numpy as np
from foundation import backbone
from scipy.interpolate import interp1d
from scipy.ndimage import map_coordinates


def straighten(
    source_image,
    spline,
    straightened_width,
):
    """Straighten a source image based on a spline fitted to the image's backbone."""

    # Adjust straightened_width
    straightened_width = straightened_width - 1

    ypoints = spline[:, 0]
    xpoints = spline[:, 1]

    x2 = xpoints[0]
    y2 = ypoints[0]

    tempinterpim = np.zeros((straightened_width + 1, len(xpoints)))

    pos = np.zeros(len(xpoints))
    pos[0] = 0
    for i in range(1, len(xpoints)):
        x1 = x2
        y1 = y2
        x2 = xpoints[i]
        y2 = ypoints[i]

        dlx = x2 - x1
        dly = y1 - y2
        le = np.sqrt(dlx * dlx + dly * dly)
        dx = dly / le
        dy = dlx / le

        if dx == 0:
            xeval = np.full((straightened_width + 1,), x1)
        else:
            xeval = np.linspace(
                x1 - (dx * straightened_width / 2),
                x1 + (dx * straightened_width / 2) + dx,
                straightened_width + 1,
            )

        if dy == 0:
            yeval = np.full((straightened_width + 1,), y1)
        else:
            yeval = np.linspace(
                y1 - (dy * straightened_width / 2),
                y1 + (dy * straightened_width / 2) + dy,
                straightened_width + 1,
            )

        coords = np.vstack((yeval, xeval))
        zeval = map_coordinates(source_image, coords, order=1)

        tempinterpim[:, i] = zeval
        if i > 0:
            pos[i] = pos[i - 1] + le

    straightened = np.zeros((tempinterpim.shape[0], int(np.ceil(pos[-1]))))
    interp_func = interp1d(
        pos, tempinterpim, kind="linear", axis=1, bounds_error=False, fill_value=0
    )
    temp = interp_func(np.arange(1, straightened.shape[1] + 1))
    straightened[:, :] = temp

    return straightened.astype(int)


def straighten_image_from_backbone(
    source_image,
    backbone,
    straightened_width,
    upsampling=2,
):
    """Straighten an image from its backbone."""

    # Get the coordinates of the backbone's longest shortest path.
    main_path_coordinates = backbone.backbone_to_longest_shortest_path(backbone)
    # Fit a B-spline to the backbone's coordinates and return the spline's coordinates.
    spline = backbone.spline_from_backbone_coordinates(
        main_path_coordinates, upsampling
    )
    straightened_image = straighten(source_image, spline, straightened_width)

    return straightened_image


def straighten_image_from_mask(
    source_image,
    mask,
    straightened_width,
    upsampling=2,
):
    """Straighten an image from its binary mask."""

    # Skeletonize the mask.
    skeleton = backbone.skeletonize_and_skan(mask)

    # Get the coordinates of the skeleton's longest shortest path.
    main_path_coordinates = backbone.skeleton_to_longest_shortest_path(skeleton)
    # Fit a B-spline to the backbone's coordinates and return the spline's coordinates.
    spline = backbone.spline_from_backbone_coordinates(
        main_path_coordinates, upsampling
    )

    straightened_image = straighten(source_image, spline, straightened_width)

    return straightened_image


def straighten_video_from_backbone(
    source_video,
    backbone_video,
    width,
    upsampling=2,
):
    """Straighten a video from a corresponding video of its backbone."""

    straightened_frames = []
    previous_backbone_tips = None
    for source_frame, backbone_frame in zip(source_video, backbone_video):  # type: ignore
        # Get the coordinates of the backbone's longest shortest path.
        try:
            backbone_coordinates = backbone.backbone_to_longest_shortest_path(
                backbone_frame
            )
        except (IndexError, ValueError):
            straightened_frame = np.zeros((width, 1))
            straightened_frames.append(straightened_frame)
            continue

        # Get the coordinates of the tips of the backbone's longuest shortest path.
        backbone_tips = backbone_coordinates[[0, -1]]
        # Check if the backbone is flipped compared to the previous frame.
        # If it is flip the backbone so that the orientation stays consistent. Update the reference tips.
        # If this frame is the first, set the reference tips.
        if previous_backbone_tips is not None:
            flipped = backbone.are_tips_flipped(previous_backbone_tips, backbone_tips)
            if flipped:
                backbone_coordinates = np.flip(backbone_coordinates, axis=0)
                backbone_tips = backbone_coordinates[[0, -1]]
            previous_backbone_tips = backbone_tips
        else:
            previous_backbone_tips = backbone_tips

        # Fit a B-spline to the backbone's coordinates and return the spline's coordinates.
        spline = backbone.spline_from_backbone_coordinates(
            backbone_coordinates, upsampling
        )
        straightened_frame = straighten(source_frame, spline, width)
        straightened_frames.append(straightened_frame)

    # Find the biggest X and Y dimensions in the frame list.
    max_x = max(straightened_frames, key=lambda x: x.shape[0]).shape[0]
    max_y = max(straightened_frames, key=lambda x: x.shape[1]).shape[1]

    video_shape = (len(straightened_frames), max_x, max_y)
    straightened_video = np.zeros(video_shape, "uint16")

    # Place the straightened frames into the video array.
    for i, frame in enumerate(straightened_frames):
        straightened_video[i, 0 : frame.shape[0], 0 : frame.shape[1]] = frame
    return straightened_video
