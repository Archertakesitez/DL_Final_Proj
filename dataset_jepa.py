from typing import NamedTuple, Optional
import torch
import numpy as np
from torchvision import transforms


# Define augmentations for states

class Augmentation:
    def __init__(self):
        # Define augmentations
        self.augmentations = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
        ])

    def __call__(self, state):
        """
        Apply augmentations to a single state (2-channel image).

        Args:
            state (torch.Tensor): Image tensor of shape (2, 65, 65).

        Returns:
            torch.Tensor: Augmented image tensor of the same shape.
        """
        augmented_channels = []
        for channel in state:  # Iterate over channels
            # Convert each channel to PIL image
            channel_pil = transforms.functional.to_pil_image(channel)
            # Apply augmentations
            channel_aug = self.augmentations(channel_pil)
            # Convert back to tensor
            augmented_channels.append(transforms.functional.to_tensor(channel_aug))

        # Stack channels back together
        return torch.stack(augmented_channels).squeeze()


class WallSample(NamedTuple):
    states: torch.Tensor
    locations: torch.Tensor
    actions: torch.Tensor


class WallDataset:
    def __init__(
        self,
        data_path,
        probing=False,
        device="cuda",
        augmentation=Augmentation()
    ):
        self.device = device
        self.states = np.load(f"{data_path}/states.npy", mmap_mode="r")
        self.actions = np.load(f"{data_path}/actions.npy")

        if probing:
            self.locations = np.load(f"{data_path}/locations.npy")
        else:
            self.locations = None

        self.augmentation = augmentation

    def __len__(self):
        return len(self.states)

    def __getitem__(self, i):
        states = torch.from_numpy(self.states[i]).float().to(self.device)
        actions = torch.from_numpy(self.actions[i]).float().to(self.device)

        if self.locations is not None:
            locations = torch.from_numpy(self.locations[i]).float().to(self.device)
        else:
            locations = torch.empty(0).to(self.device)

        # Apply augmentation consistently to the entire trajectory
        if self.augmentation:
            states = self._apply_augmentation(states)

        return WallSample(states=states, locations=locations, actions=actions)
    

    def _apply_augmentation(self, states):
        """
        Apply the same augmentation to all images in the trajectory.

        Args:
            states (torch.Tensor): Tensor of shape (trajectory_length, 2, 65, 65).

        Returns:
            torch.Tensor: Augmented states of the same shape.
        """
        augmented_states = []
        for t in range(states.shape[0]):  # Loop over trajectory_length
            state = states[t]  # (2, 65, 65)
            # Convert tensor to PIL image, apply augmentation, then back to tensor
            state_aug = self.augmentation(state)
            augmented_states.append(state_aug)
        return torch.stack(augmented_states)  # Reconstruct trajectory tensor


def create_wall_dataloader(
    data_path,
    probing=False,
    device="cuda",
    batch_size=64,
    train=True,
):
    ds = WallDataset(
        data_path=data_path,
        probing=probing,
        device=device,
    )

    loader = torch.utils.data.DataLoader(
        ds,
        batch_size,
        shuffle=train,
        drop_last=True,
        pin_memory=False,
    )

    return loader