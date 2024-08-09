import yaml
import torch
import os
import numpy as np
from torch.nn import functional as F
from src_py import models
from src_py.custom_transforms import build_transform
import torchvision.transforms.functional as tv_F
from torch.utils.data import Dataset, DataLoader
import tqdm

# with open(cfg_path, "r") as f:
#     job_description = yaml.full_load(f)
# np.random.seed(seed)
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DatasetPatches_M(Dataset):
    def __init__(
        self,
        org_frames: torch.Tensor,
        stl_frames: torch.Tensor,
        stl_indices: list[int],
        msk_frames: torch.Tensor | None,
        flw_fwds: list[np.ndarray] | None,
        flw_bwds: list[np.ndarray] | None,
        patch_size: int = 2**5,
        batch_size: int = 2**2,
        patch_num: int = 2**6,
    ):
        # B C H W
        self.transform = build_transform()
        self.org_frames = org_frames
        self.stl_frames = stl_frames
        self.stl_indices = stl_indices
        self.msk_frames = msk_frames
        self.flw_fwds = flw_fwds
        self.flw_bwds = flw_bwds

        self.patch_size = patch_size
        self.batch_size = batch_size
        self.patch_num = patch_num

        self.H = self.org_frames.shape[2]
        self.W = self.org_frames.shape[3]

        self.len_stl = len(self.stl_frames)
        assert (
            self.len_stl == len(self.stl_indices)
        ), f"Expected style indices == style frames. Got {len(self.stl_indices)=}, {self.len_stl=}"

        self.len_imgs = len(self.stl_frames)
        assert (
            self.len_imgs >= self.len_stl
        ), f"Got less org than style. {self.len_imgs=}, {self.len_stl=}"

    def build_gauss(self):
        pass

    def __len__(self):
        return self.len_imgs

    def __getitem__(self, item):
        # print(f"{item=}")
        img_index = item % self.len_stl
        # print(f"{self.org_frames[self.stl_indices[img_index]].shape=}")

        indices = get_valid_indices(
            self.H, self.W, self.patch_size, self.patch_size // 4
        )

        input_patches = cut_patches(
            self.org_frames[self.stl_indices[img_index]], indices, self.patch_size, 0
        )
        gt_patches = cut_patches(
            self.stl_frames[img_index], indices, self.patch_size, 0
        )
        # TODO: Gauss patches

        return input_patches, gt_patches


# def np_to_tensor(seq: list[np.ndarray]):
#     # [H, W, C]
#     ts = [torch.from_numpy(se / 255.0).permute(2, 0, 1).unsqueeze(0) for se in seq]
#     # [1, C, H, W]
#     tensor = torch.cat(ts, dim=0).float()  # [B, C, H, W]

#     return tensor


def np_to_tensor(seq: list[np.ndarray]):
    # Convert from BGR to RGB
    rgb_seq = [np.ascontiguousarray(se[..., ::-1]) for se in seq]

    # Normalize to [0, 1] and convert to tensor
    ts = [torch.from_numpy(se / 255.0).permute(2, 0, 1).unsqueeze(0) for se in rgb_seq]

    # Concatenate along the batch dimension
    tensor = torch.cat(ts, dim=0).float()  # [B, C, H, W]

    return tensor


def np_msk_to_tensor(seq: list[np.ndarray]):
    # [H, W]
    ts = [torch.from_numpy(se / 255.0).unsqueeze(0) for se in seq]
    # [1, H, W]
    tensor = torch.cat(ts, dim=0).float()  # [B, H, W]

    return tensor


def init_model(job_description: dict, device):
    cfg = job_description["job"]

    generator, opt_generator = get_generator(cfg, device)

    discriminator, opt_discriminator = get_discriminator(cfg, device)

    perception_loss_model, perception_loss_weight = get_perception_loss(cfg, device)
    if perception_loss_model is None:
        perception_loss_model = discriminator

    reconstruction_criterion, adversarial_criterion = get_criterion(cfg)

    return (
        generator,
        opt_generator,
        discriminator,
        opt_discriminator,
        perception_loss_model,
        perception_loss_weight,
        reconstruction_criterion,
        adversarial_criterion,
    )


def train(
    org_frames: torch.Tensor,
    stl_frames: torch.Tensor,
    stl_indices: list[int],
    generator: models.GeneratorJ,
    opt_generator: torch.optim.Adam,
    discriminator: models.DiscriminatorN_IN | None,
    opt_discriminator: torch.optim.Adam,
    perception_loss_model: models.PerceptualVGG19,
    reconstruction_criterion: torch.nn.L1Loss,
    adversarial_criterion: torch.nn.MSELoss,
    device,
    num_epochs=50_000_000,
    seed=0,
    perception_loss_weight=6.0,
    reconstruction_weight=4.0,
    adversarial_weight=0.5,
    batch_size=32,
    patch_size=32,
):
    train_dataset = DatasetPatches_M(
        org_frames=org_frames,
        stl_frames=stl_frames,
        stl_indices=stl_indices,
        msk_frames=None,
        flw_fwds=None,
        flw_bwds=None,
        patch_size=patch_size,
    )

    gen = torch.Generator()
    gen.manual_seed(seed)

    patch_loader = DataLoader(train_dataset, 1, shuffle=True)

    last_loss = 0.0
    last_g_losses = []
    last_d_losses = []

    generator.train()
    perception_loss_model.train()

    use_adversarial_loss = False
    if discriminator is not None:
        use_adversarial_loss = True
        discriminator.train()

    for epoch in (pbar := tqdm.tqdm(range(num_epochs + 1))):
        g_losses = []
        d_losses = []
        for i, (input_patches, gt_patches) in enumerate(patch_loader):
            # Get a pair of Original-Styled
            input_patches, gt_patches = (
                input_patches.squeeze().to(device),
                gt_patches.squeeze().to(device),
            )

            # Split the tensors into batches of size 16
            input_batches = torch.split(input_patches, batch_size)
            gt_batches = torch.split(gt_patches, batch_size)

            for input_batch, gt_batch in zip(input_batches, gt_batches):
                image_loss = 0
                perception_loss = 0
                adversarial_loss = 0

                # Train discriminator
                if use_adversarial_loss:
                    assert discriminator is not None
                    opt_discriminator.zero_grad()

                    d_generated = generator.forward(input_batch)

                    # how well can it label as fake?
                    fake_pred = discriminator(d_generated)
                    fake = torch.zeros_like(fake_pred)
                    fake_loss = adversarial_criterion.forward(fake_pred, fake)

                    # how well can it label as real?
                    real_pred = discriminator.forward(gt_batch)
                    real = torch.ones_like(real_pred)

                    real_loss = adversarial_criterion.forward(real_pred, real)

                    discriminator_loss = real_loss + fake_loss
                    discriminator_loss.backward()
                    opt_discriminator.step()

                    d_losses.append(discriminator_loss.item())

                # Train generator
                opt_generator.zero_grad()
                generated = generator(input_batch)

                ## Image Loss
                image_loss = reconstruction_criterion.forward(generated, gt_batch)

                ## Perceptual Loss
                fake_features = perception_loss_model.forward(generated)
                target_features = perception_loss_model.forward(gt_batch)
                perception_loss = ((fake_features - target_features) ** 2).mean()

                generator_loss = (
                    reconstruction_weight * image_loss
                    + perception_loss_weight * perception_loss
                )

                if use_adversarial_loss:
                    # How good the generator is at fooling the discriminator
                    fake_pred = discriminator(generated)
                    adversarial_loss = adversarial_criterion(
                        fake_pred, torch.ones_like(fake_pred)
                    )
                    generator_loss += adversarial_weight * adversarial_loss

                generator_loss.backward()
                opt_generator.step()

                g_losses.append(generator_loss.item())

        last_loss = np.mean(g_losses)
        d_last_loss = np.mean(d_losses)
        
        last_g_losses.append(last_loss)
        last_d_losses.append(d_last_loss)
        
        pbar.set_description(
            f"Training epoch [{epoch}] G[{last_loss:.5f}] D[{d_last_loss:.5f}]"
        )
    
    return last_g_losses, last_d_losses


def get_generator(cfg: dict, device):
    generator = build_model(cfg["generator"]["type"], cfg["generator"]["args"], device)

    opt_generator = build_optimizer(
        cfg["opt_generator"]["type"], generator, cfg["opt_generator"]["args"]
    )

    return generator, opt_generator


def get_discriminator(cfg: dict, device):
    discriminator, opt_discriminator = None, None
    if "discriminator" in cfg:
        discriminator = build_model(
            cfg["discriminator"]["type"], cfg["discriminator"]["args"], device
        )
        opt_discriminator = build_optimizer(
            cfg["opt_discriminator"]["type"],
            discriminator,
            cfg["opt_discriminator"]["args"],
        )
    return discriminator, opt_discriminator


def get_perception_loss(cfg: dict, device):
    perception_loss_model = None
    perception_loss_weight = 1
    if "perception_loss" in cfg:
        if "perception_model" in cfg["perception_loss"]:
            perception_loss_model = build_model(
                cfg["perception_loss"]["perception_model"]["type"],
                cfg["perception_loss"]["perception_model"]["args"],
                device,
            )

        perception_loss_weight = cfg["perception_loss"]["weight"]

    return perception_loss_model, perception_loss_weight


def get_criterion(cfg: dict):
    reconstruction_criterion = getattr(
        torch.nn, cfg["trainer"]["reconstruction_criterion"]
    )()
    adversarial_criterion = getattr(torch.nn, cfg["trainer"]["adversarial_criterion"])()

    return reconstruction_criterion, adversarial_criterion


def build_model(model_type, args, device):
    assert model_type in ["GeneratorJ", "DiscriminatorN_IN", "PerceptualVGG19"]
    model = getattr(models, model_type)(**args)
    return model.to(device)


def build_optimizer(opt_type, model, args):
    assert opt_type in ["Adam"]
    args["params"] = model.parameters()
    opt_class = getattr(torch.optim, opt_type)
    return opt_class(**args)


def get_valid_indices(H: int, W: int, patch_size: int, random_overlap: int = 0):
    vih = torch.arange(random_overlap, H - patch_size - random_overlap + 1, patch_size)
    viw = torch.arange(random_overlap, W - patch_size - random_overlap + 1, patch_size)
    if random_overlap > 0:
        rih = torch.randint_like(vih, -random_overlap, random_overlap)
        riw = torch.randint_like(viw, -random_overlap, random_overlap)
        vih += rih
        viw += riw
    vi = torch.stack(torch.meshgrid(vih, viw, indexing="ij")).view(2, -1).t()  # (N, 2)
    return vi


def cut_patches(
    inp: torch.Tensor,
    indices: torch.Tensor,
    patch_size: int,
    padding: int = 0,
):
    # input: [C, H, W]
    patches = []

    # Calculate patch dimensions
    patch_dim = patch_size + 2 * padding

    # Create a tensor of offsets for each pixel in the patch
    y_offsets = torch.arange(-padding, patch_dim - padding, device=inp.device)
    x_offsets = torch.arange(-padding, patch_dim - padding, device=inp.device)

    # Create a grid of all pixel coordinates for each patch
    y_indices = indices[:, 0, None, None] + y_offsets[None, :, None]
    x_indices = indices[:, 1, None, None] + x_offsets[None, None, :]

    # Clamp indices to ensure they're within the input tensor bounds
    y_indices = y_indices.clamp(0, inp.shape[-2] - 1)
    x_indices = x_indices.clamp(0, inp.shape[-1] - 1)

    # Extract all patches at once
    patches = inp[:, y_indices, x_indices]  # [C, N, patch_dim, patch_dim]
    patches = patches.permute(1, 0, 2, 3)  # [N, C, patch_dim, patch_dim]

    return patches


def reconstruct_image(
    patches: torch.Tensor,
    indices: torch.Tensor,
    H: int,
    W: int,
    patch_size: int,
    padding: int = 0,
):
    # Initialize an empty tensor for the reconstructed image
    reconstructed = torch.zeros((patches.shape[1], H, W), device=patches.device)
    count = torch.zeros((H, W), device=patches.device)

    # Calculate the effective patch size including padding
    patch_dim = patch_size + 2 * padding

    for i, (y, x) in enumerate(indices):
        # Calculate the region of the image that corresponds to the current patch
        y_slice = slice(y, y + patch_dim)
        x_slice = slice(x, x + patch_dim)

        # Add the current patch to the reconstructed image
        reconstructed[:, y_slice, x_slice] += patches[i]
        count[y_slice, x_slice] += 1

    # Normalize by the count to handle overlapping patches
    reconstructed /= count.clamp(min=1)

    # Remove padding by slicing the valid region
    if padding > 0:
        reconstructed = reconstructed[:, padding:-padding, padding:-padding]

    return reconstructed
