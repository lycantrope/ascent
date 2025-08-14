import logging
import math
import numbers
from typing import Sequence

import torch
import torch.nn.functional as F
from torch.utils.data import default_collate
from torchvision.transforms import ColorJitter
from torchvision.transforms.functional import InterpolationMode
from torchvision.transforms.transforms import _setup_angle
from torchvision.transforms.v2.functional import resize, rotate_image

from ascent.datasets.tracking_dataset import OEDItem


def process_batch_or_single(data: dict, process_func):
    """Process batched or single data using the provided processing function."""
    if data["image"].dim() == 5:
        processed_items = []
        for idx in range(data["image"].size(0)):
            processed_items.append(process_func(OEDItem.get_item(data, idx)))
        return default_collate(processed_items)
    else:
        return process_func(data)


def flow_warp(x, flow, interpolation="bilinear", padding_mode="zeros", align_corners=True):
    """Warp an image (N,C,H,W) with flow (N,H,W,2)."""
    n, c, h, w = x.size()
    device = x.device
    # create mesh grid if needed
    grid_y, grid_x = torch.meshgrid(
        torch.arange(0, h, device=device), torch.arange(0, w, device=device), indexing="ij"
    )
    grid = torch.stack((grid_x, grid_y), 2).type_as(x)  # (h,w,2)
    grid_flow = grid[None, ...] + flow  # (N,h,w,2)
    # scale grid_flow to [-1,1]
    grid_flow_x = 2.0 * grid_flow[..., 0] / max(w - 1, 1) - 1.0
    grid_flow_y = 2.0 * grid_flow[..., 1] / max(h - 1, 1) - 1.0
    grid_flow = torch.stack((grid_flow_x, grid_flow_y), dim=3)

    output = F.grid_sample(
        x, grid_flow, mode=interpolation, padding_mode=padding_mode, align_corners=align_corners
    )
    return output


def invert_flow_position_batch(flow, x_in, y_in):
    """
    Batched version of invert_flow_position:

    flow: shape (H,W,2) (backward warp flow)
    x_in, y_in: tensors of shape (B,) representing B input coordinates
    => returns (X_out, Y_out) of shape (B,) s.t pos_grid[Y_out, X_out] ~ (x_in, y_in).
    """
    H, W, _ = flow.shape
    device = flow.device

    # Build base coordinate grid
    grid_y, grid_x = torch.meshgrid(
        torch.arange(H, device=device), torch.arange(W, device=device), indexing="ij"
    )
    base_grid = torch.stack([grid_x, grid_y], dim=2)  # (H,W,2)
    pos_grid = base_grid + flow  # (H,W,2)

    # Flatten pos_grid => shape (H*W,2)
    pos_flat = pos_grid.view(-1, 2)  # (H*W,2)

    # Expand input coordinates to shape (B, H*W) for broadcasting
    x_in = x_in[:, None]  # (B,1)
    y_in = y_in[:, None]  # (B,1)

    # Compute squared distance for all points in batch
    dx = pos_flat[:, 0].unsqueeze(0) - x_in  # (B, H*W)
    dy = pos_flat[:, 1].unsqueeze(0) - y_in  # (B, H*W)
    dist_sq = dx * dx + dy * dy  # (B, H*W)

    # Get indices of minimum distance per batch item
    min_idx = torch.argmin(dist_sq, dim=1)  # (B,)

    # Convert min_idx back to (Y*, X*)
    out_y = min_idx // W
    out_x = min_idx % W

    return out_x.float(), out_y.float()


def rotate_points_batch(points, angle, center):
    """
    Rotates a batch of 2D points around a center by a given angle in degrees.

    Args:
        points (torch.Tensor): Tensor of shape (N, 2), where N is the number of points.
        angle (float): Rotation angle in degrees.
        center (torch.Tensor): Tensor of shape (2,) representing the center of rotation.

    Returns:
        torch.Tensor: Tensor of shape (N, 2) containing the rotated points.
    """
    assert points.dim() == 2 and points.size(1) == 2, (
        "Input points must be a tensor of shape (N, 2)."
    )
    # Convert angle to radians
    angle_rad = angle * math.pi / 180.0

    # Translate points to the origin
    translated_points = points - center

    # Rotation matrix
    c = torch.cos(angle_rad)
    s = torch.sin(angle_rad)
    rotation_matrix = torch.tensor([[c, -s], [s, c]], dtype=points.dtype, device=points.device)

    # Rotate points
    rotated_points = torch.matmul(translated_points, rotation_matrix.T)

    # Translate points back
    rotated_points += center

    return rotated_points


class OEDRandomRotation3D_XYplane:
    def __init__(
        self,
        degrees: numbers.Number | Sequence,
        interpolation=InterpolationMode.BILINEAR,
        expand=False,
        center=None,
        fill=None,
        allow_dropout=False,
        max_attempts=10,
    ):
        """
        Rotate the input by angle.

        Args:
            degrees (sequence or number): Range of degrees to select from.
                If degrees is a number instead of sequence like (min, max), the range of degrees
                will be (-degrees, +degrees).
            interpolation (InterpolationMode, optional): Desired interpolation enum defined by
                :class:`torchvision.transforms.InterpolationMode`. Default is ``InterpolationMode.NEAREST``.
                If input is Tensor, only ``InterpolationMode.NEAREST``, ``InterpolationMode.BILINEAR`` are supported.
                The corresponding Pillow integer constants, e.g. ``PIL.Image.BILINEAR`` are accepted as well.
            expand (bool, optional): Optional expansion flag.
                If true, expands the output to make it large enough to hold the entire rotated image.
                If false or omitted, make the output image the same size as the input image.
                Note that the expand flag assumes rotation around the center (see note below) and no translation.
            center (sequence, optional): Optional center of rotation, (x, y). Origin is the upper left corner.
                Default is the center of the image.

                .. note::

                    In theory, setting ``center`` has no effect if ``expand=True``, since the image center will become the
                    center of rotation. In practice however, due to numerical precision, this can lead to off-by-one
                    differences of the resulting image size compared to using the image center in the first place. Thus, when
                    setting ``expand=True``, it's best to leave ``center=None`` (default).
            fill (number or tuple or dict, optional): Pixel fill value used when the  ``padding_mode`` is constant.
                Default is 0. If a tuple of length 3, it is used to fill R, G, B channels respectively.
                Fill value can be also a dictionary mapping data type to the fill value, e.g.
                ``fill={tv_tensors.Image: 127, tv_tensors.Mask: 0}`` where ``Image`` will be filled with 127 and
                ``Mask`` will be filled with 0.
        """
        self.degrees = _setup_angle(degrees, name="degrees", req_sizes=(2,))
        self.interpolation = interpolation
        self.expand = expand
        self.center = center
        self.fill = fill
        self.allow_dropout = allow_dropout
        self.max_attempts = max_attempts

    def process_single_item(self, data: OEDItem) -> OEDItem:
        image = data["image"]
        z = data["coords"][:, -3].clone()
        y = data["coords"][:, -2]
        x = data["coords"][:, -1]
        new_object_ids = data["object_ids"].clone()

        for _ in range(self.max_attempts):
            angle = torch.empty(1, device=image.device).uniform_(self.degrees[0], self.degrees[1])[
                0
            ]
            if image.size(0) == 0:
                # empty image - PE only mode
                new_image = image
            else:
                new_image = rotate_image(
                    image, angle, self.interpolation, self.expand, self.center, fill=self.fill
                )
            if self.center is None:
                center = torch.tensor(
                    [image.shape[-2] / 2, image.shape[-1] / 2], device=image.device
                )
            else:
                center = torch.tensor(self.center, device=image.device)

            new_points = rotate_points_batch(torch.stack([y, x], dim=-1), angle, center)
            new_y = new_points[:, 0]
            new_x = new_points[:, 1]
            b_dropout = (
                (new_y < 0) | (new_y >= image.shape[-2]) | (new_x < 0) | (new_x >= image.shape[-1])
            )
            b_dropout_prev = new_object_ids < 0
            # make sure the padded pseudo objects remain inactive
            b_dropout_new = b_dropout | b_dropout_prev
            if self.allow_dropout:
                # dropout object outside the image
                new_y[b_dropout_new] = -1
                new_x[b_dropout_new] = -1
                z[b_dropout_new] = -1
                new_object_ids[b_dropout_new] = -1
                break
            else:
                # if there's no additional dropout, we can break
                if b_dropout_new.sum() == b_dropout_prev.sum():
                    break
                else:
                    logging.debug(
                        "Random rotation failed. Retrying with a new random angle. This is attempt %d.",
                    )
        else:
            logging.debug("Random rotation failed. Use the original image.")
            new_image = image
            new_x = x
            new_y = y

        return data.new_item(
            image=new_image,
            object_ids=new_object_ids,
            coords=torch.stack([z, new_y, new_x], dim=-1),
        )

    def __call__(self, data: dict) -> dict:
        if self.degrees[0] == 0 and self.degrees[1] == 0:
            # No rotation applied
            return data
        else:
            return process_batch_or_single(data, self.process_single_item)


class OEDRandomResizedCropXY:
    """
    Randomly resize and crop the input image to a target size.
    """

    def __init__(
        self,
        scale_x: tuple[float, float] = (0.9, 1.1),
        scale_y: tuple[float, float] = (0.9, 1.1),
        interpolation: InterpolationMode | int = InterpolationMode.BILINEAR,
        antialias: bool | None = True,
        crop_size: int | tuple[int, int] = (256, 256),
        allow_dropout: bool = False,
        max_attempts: int = 10,
    ):
        self.scale_x = scale_x
        self.scale_y = scale_y
        self.interpolation = interpolation
        self.antialias = antialias
        if isinstance(crop_size, int):
            self.crop_size = (crop_size, crop_size)
        else:
            self.crop_size = crop_size
        self.allow_dropout = allow_dropout
        self.max_attempts = max_attempts

    def process_single_item(self, data: OEDItem) -> OEDItem:
        image = data["image"]
        z = data["coords"][:, -3].clone()
        y = data["coords"][..., -2]
        x = data["coords"][..., -1]
        new_object_ids = data["object_ids"].clone()
        for _ in range(self.max_attempts):
            # Resize the image
            size_range_x = (
                int(image.shape[-1] * self.scale_x[0]),
                int(image.shape[-1] * self.scale_x[1]) + 1,
            )
            size_range_y = (
                int(image.shape[-2] * self.scale_y[0]),
                int(image.shape[-2] * self.scale_y[1]) + 1,
            )
            size_x = int(torch.randint(size_range_x[0], size_range_x[1], ()))
            size_y = int(torch.randint(size_range_y[0], size_range_y[1], ()))
            if image.size(0) == 0:
                # empty image - PE only mode
                new_image = torch.empty(0, image.size(1), size_y, size_x, device=image.device)
            else:
                new_image = resize(image, (size_y, size_x), self.interpolation)
            new_y = y * size_y / image.shape[-2]
            new_x = x * size_x / image.shape[-1]

            # Random crop the image
            assert size_y >= self.crop_size[0] and size_x >= self.crop_size[1], (
                f"Cannot crop to size {self.crop_size}. Image size is ({size_y}, {size_x})."
            )
            crop_y = int(torch.randint(0, size_y - self.crop_size[0] + 1, ()))
            crop_x = int(torch.randint(0, size_x - self.crop_size[1] + 1, ()))
            new_image = new_image[
                ..., crop_y : crop_y + self.crop_size[0], crop_x : crop_x + self.crop_size[1]
            ]
            new_y = new_y - crop_y
            new_x = new_x - crop_x

            b_dropout = (
                (new_y < 0)
                | (new_y >= self.crop_size[0])
                | (new_x < 0)
                | (new_x >= self.crop_size[1])
            )
            b_dropout_prev = new_object_ids < 0
            # make sure the padded pseudo objects remain inactive
            b_dropout_new = b_dropout | b_dropout_prev

            if self.allow_dropout:
                # dropout object outside the image
                new_y[b_dropout_new] = -1
                new_x[b_dropout_new] = -1
                z[b_dropout_new] = -1
                new_object_ids[b_dropout_new] = -1
                break
            else:
                # if there's no additional dropout, we can break
                if b_dropout_new.sum() == b_dropout_prev.sum():
                    break
                else:
                    logging.debug(
                        "Random resized crop failed. Retrying with a new random size. This is attempt %d.",
                    )
        else:
            logging.warning("Random resized crop failed. Use the original image.")
            new_image = image
            new_x = data["coords"][..., -1]
            new_y = data["coords"][..., -2]

        return data.new_item(
            image=new_image,
            object_ids=new_object_ids,
            coords=torch.stack([z, new_y, new_x], dim=-1),
        )

    def __call__(self, data: dict) -> dict:
        return process_batch_or_single(data, self.process_single_item)


class OEDColorJitter3D:
    """
    Apply ColorJitter transformation separately to each channel of the image tensor in the sample,
    excluding any one-hot encoded positional channels. This is particularly useful for microscopy images where each channel
    might represent different fluorescent tags, and it is essential to adjust the visual properties of each channel independently
    to enhance feature visibility without distorting positional encoding.

    This transformation applies adjustments such as brightness and contrast independently to each channel, which can help in
    situations where different channels have different lighting conditions or contrast characteristics.
    """

    def __init__(self, brightness=0, contrast=0):
        """
        Initialize the color jitter transformation with the given parameters.

        Args:
            brightness (float or tuple of float (min, max)): How much to jitter brightness.
                brightness_factor is chosen uniformly from [max(0, 1 - brightness), 1 + brightness]
                or the given [min, max]. Should be non negative numbers.
            contrast (float or tuple of float (min, max)): How much to jitter contrast.
                contrast_factor is chosen uniformly from [max(0, 1 - contrast), 1 + contrast]
                or the given [min, max]. Should be non negative numbers.
        """
        self.color_jitter = ColorJitter(
            brightness=brightness, contrast=contrast, saturation=0, hue=0
        )

    def process_single_item(self, data: OEDItem) -> OEDItem:
        image = data["image"]
        if image.size(0) > 0:
            transformed_image = image.clone()
            for c in range(image.size(0)):
                image_temp = image[c : c + 1].permute(1, 0, 2, 3)
                transformed_image[c : c + 1] = self.color_jitter(image_temp).permute(1, 0, 2, 3)
            return data.new_item(image=transformed_image)
        else:
            return data

    def __call__(self, data: dict) -> dict:
        return process_batch_or_single(data, self.process_single_item)


class OEDObjectPositionJitter3D:
    """
    Apply random jitter to the object position in 3D images. This transformation is useful for data augmentation
    to simulate the variability in object positions in the training data. The jitter is applied independently to
    the x, y, and z coordinates of the object position.

    Args:
        xy_jitter_range (float or tuple of float): The range of jitter to apply to the x and y coordinates.
            If a float, the same range is applied to both x and y coordinates.
            If a tuple, the first element is the range for the x coordinate and the second element is the range for the y coordinate.
        z_jitter_range (float): The range of jitter to apply to the z coordinate.
    """

    def __init__(self, xy_jitter_range: float | tuple[float, float], z_jitter_range: float):
        if isinstance(xy_jitter_range, numbers.Number):
            self.xy_jitter_range = (xy_jitter_range, xy_jitter_range)
        else:
            self.xy_jitter_range = xy_jitter_range
        self.z_jitter_range = z_jitter_range

    def process_single_item(self, data: OEDItem) -> OEDItem:
        device = data["image"].device
        n_samples = data["coords"].size(0)
        object_ids = data["object_ids"]
        jitter_x = torch.empty(n_samples, device=device).uniform_(
            -self.xy_jitter_range[0], self.xy_jitter_range[0]
        )
        jitter_y = torch.empty(n_samples, device=device).uniform_(
            -self.xy_jitter_range[1], self.xy_jitter_range[1]
        )
        jitter_z = torch.empty(n_samples, device=device).uniform_(
            -self.z_jitter_range, self.z_jitter_range
        )
        new_x = torch.clip(data["coords"][:, -1] + jitter_x, 0, data["image"].size(-1) - 1)
        new_y = torch.clip(data["coords"][:, -2] + jitter_y, 0, data["image"].size(-2) - 1)
        new_z = torch.clip(data["coords"][:, -3] + jitter_z, 0, data["image"].size(-3) - 1)
        new_x[object_ids < 0] = -1
        new_y[object_ids < 0] = -1
        new_z[object_ids < 0] = -1
        return data.new_item(coords=torch.stack([new_z, new_y, new_x], dim=-1))

    def __call__(self, data: dict) -> dict:
        return process_batch_or_single(data, self.process_single_item)


class OEDElasticDeformRandomGrid2Dfor3D:
    """
    GPU-accelerated version of OEDElasticDeformRandomGrid2Dfor3D.

    We:
    1. Generate a random displacement field on GPU.
    2. Apply flow_warp to each Z-slice.
    3. Update object coordinates by sampling flow at their location.

    Args:
        sigma (float): Standard deviation of the random displacements.
        points (int): Number of control points along H and W to define the coarse displacement grid.
        mode (str): Padding mode for grid_sample ('zeros', 'border', 'reflection').
        order (int): Not directly used since we rely on grid_sample. (We can ignore or adapt interpolation.)
    """

    def __init__(
        self,
        sigma: float,
        points: int,
        mode: str = "border",
        order: int = 1,
        allow_dropout: bool = False,
        max_attempts: int = 10,
    ):
        self.sigma = sigma
        self.points = points
        self.mode = mode
        self.order = (
            order  # For bilinear/bicubic interpolation we just fix it to 'bilinear' in flow_warp
        )
        self.allow_dropout = allow_dropout
        self.max_attempts = max_attempts  # Number of retries before forcing a solution

    def generate_displacement_field(self, H, W, device):
        """
        Generate a random displacement field for the XY plane:
        1. Create a coarse grid of control points.
        2. Sample random displacements at control points.
        3. Upsample to full resolution using bicubic or bilinear interpolation.

        Returns:
            flow (Tensor): shape (1,H,W,2) containing displacement in X and Y.
        """
        y_lin = torch.linspace(0, H - 1, self.points, device=device)
        x_lin = torch.linspace(0, W - 1, self.points, device=device)
        Yc, Xc = torch.meshgrid(y_lin, x_lin, indexing="ij")

        disp_y = torch.randn_like(Yc) * self.sigma
        disp_x = torch.randn_like(Xc) * self.sigma

        # Interpolate to full size
        disp_y = disp_y.unsqueeze(0).unsqueeze(0)  # (1,1,points,points)
        disp_x = disp_x.unsqueeze(0).unsqueeze(0)

        # Use bicubic or bilinear interpolation
        mode = "bicubic" if self.order > 1 else "bilinear"
        disp_y = F.interpolate(disp_y, size=(H, W), mode=mode, align_corners=True)
        disp_x = F.interpolate(disp_x, size=(H, W), mode=mode, align_corners=True)

        # Stack into (1,H,W,2)
        flow = torch.cat(
            [disp_x.squeeze(0).permute(1, 2, 0), disp_y.squeeze(0).permute(1, 2, 0)], dim=2
        )
        # Now flow is (H,W,2), add batch dimension
        flow = flow.unsqueeze(0)  # (1,H,W,2)
        return flow

    def process_single_item(self, data: OEDItem) -> OEDItem:
        """
        Apply the GPU-based deformation.
        """
        image = data["image"]
        coords = data["coords"]
        new_object_ids = data["object_ids"].clone()
        device = image.device
        C, D, H, W = image.shape
        new_coords = coords.clone()

        for attempt in range(self.max_attempts):
            flow = self.generate_displacement_field(H, W, device)  # (1,H,W,2)

            if image.size(0) == 0:
                new_image = image
            else:
                # Warp all slices in one go:
                # image: (C,D,H,W)
                # Treat D dimension as batch: (D,C,H,W)
                image_for_warp = image.permute(1, 0, 2, 3)  # (D,C,H,W)
                # Expand flow to (D,H,W,2) for all slices
                flow_expanded = flow.expand(image_for_warp.size(0), H, W, 2)  # (D,H,W,2)

                warped = flow_warp(
                    image_for_warp, flow_expanded, interpolation="bilinear", padding_mode=self.mode
                )
                # warped: (D,C,H,W)
                new_image = warped.permute(1, 0, 2, 3)  # (C,D,H,W)

            # Update point coordinates
            new_x, new_y = invert_flow_position_batch(flow[0], coords[:, -1], coords[:, -2])
            new_coords[:, -1] = new_x
            new_coords[:, -2] = new_y
            b_dropout = (new_x < 0) | (new_x >= W) | (new_y < 0) | (new_y >= H)
            b_dropout_prev = new_object_ids < 0
            # make sure the padded pseudo objects remain inactive
            b_dropout_new = b_dropout | b_dropout_prev
            if self.allow_dropout:
                # dropout object outside the image
                new_coords[b_dropout_new] = -1
                new_object_ids[b_dropout_new] = -1
                break
            else:
                # if there's no additional dropout, we can break
                if b_dropout_new.sum() == b_dropout_prev.sum():
                    break
                else:
                    logging.debug(
                        "Elastic deformation failed. Retrying with a new random displacement field. This is attempt %d.",
                    )
        else:
            # Fall back to no deformation.
            logging.warning(
                f"Deformation failed after {self.max_attempts} attempts. Skipping deformation."
            )
            return data

        return data.new_item(image=new_image, object_ids=new_object_ids, coords=new_coords)

    def __call__(self, data: dict) -> dict:
        if self.sigma == 0:
            return data
        else:
            return process_batch_or_single(data, self.process_single_item)


class OEDObjectDropout:
    """
    Randomly drop objects from the input data. This transformation is useful for data augmentation to simulate
    scenarios where objects are not detected or are missed in the input data. The dropout is applied independently
    to each object in the input data.

    Dropout is applied by setting the object id and the corresponding coordinates to -1.

    Args:
        p (float): The probability of dropping an object.
    """

    def __init__(self, p: float):
        self.p = p

    def process_single_item(self, data: OEDItem) -> OEDItem:
        object_ids = data["object_ids"]
        dropout_mask = torch.rand_like(object_ids, dtype=torch.float32) < self.p
        new_object_ids = object_ids.clone()
        new_object_ids[dropout_mask] = -1
        new_coords = data["coords"].clone()
        new_coords[dropout_mask] = -1
        return data.new_item(object_ids=new_object_ids, coords=new_coords)

    def __call__(self, data: dict) -> dict:
        return process_batch_or_single(data, self.process_single_item)


if __name__ == "__main__":
    pass
