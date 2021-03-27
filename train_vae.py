"""Code adapted from lucidrains/DALLE-pytorch
Original GitHub link: https://github.com/lucidrains/DALLE-pytorch
"""
# Import built-in package
import logging
import os
import math
from math import sqrt
import argparse

# Import pytorch package
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision.utils import make_grid

# Import DALLE pytorch
from dalle_pytorch import DiscreteVAE

# Import pillow for process image
from PIL import Image

# Set to info level for prompting in terminal
from tqdm import tqdm

# Weight and bias for seeing the result
import wandb

# Set up logger
logging.getLogger().setLevel(logging.INFO)

# Argument parsing
parser = argparse.ArgumentParser()
parser.add_argument('--image_folder',
                    type=str,
                    required=True,
                    help='Path to your folder of images for learning the discrete VAE and its codebook')
parser.add_argument('--image_size',
                    type=int,
                    required=False,
                    default=128,
                    help='Image size to be reshaped')
args = parser.parse_args()

# Constants
IMAGE_SIZE = args.image_size
IMAGE_PATH = args.image_folder

# Hyperparameters for training
EPOCHS = 20
BATCH_SIZE = 16
LEARNING_RATE = 1e-3
LR_DECAY_RATE = 0.98

# Hyperparameters for models
NUM_TOKENS = 8192
NUM_LAYERS = 2
NUM_RESNET_BLOCKS = 2
SMOOTH_L1_LOSS = False
EMB_DIM = 512
HID_DIM = 256
KL_LOSS_WEIGHT = 0

STARTING_TEMP = 1.
TEMP_MIN = 0.5
ANNEAL_RATE = 1e-6

NUM_IMAGES_SAVE = 4

# Check cuda is available or not
device = "cuda:0" if torch.cuda.is_available() else "cpu"

model_config = dict(
    num_tokens=NUM_TOKENS,
    smooth_l1_loss=SMOOTH_L1_LOSS,
    num_resnet_blocks=NUM_RESNET_BLOCKS,
    kl_loss_weight=KL_LOSS_WEIGHT
)

run = wandb.init(
    project='dalle_train_vae',
    job_type='train_model',
    config=model_config
)


class TrainDataset(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        files_under_root = os.listdir(root)

        # Find all image under folder
        IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png',
                          '.ppm', '.bmp', '.pgm', '.tif',
                          '.tiff', '.webp')
        self.images = []
        for file in files_under_root:
            if file.lower().endswith(IMG_EXTENSIONS):
                self.images.append(file)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, item):
        # Read in the image
        img_path = os.path.join(self.root, self.images[item])
        cur_img = Image.open(img_path).convert("RGB")

        # Check the transform
        if self.transform is not None:
            cur_img = self.transform(cur_img)

        # Return cur_img, item is the dummy variable which will
        #  be omitted latter
        return cur_img, item


# Creat dataset and dataloader
ds = TrainDataset(
    root=IMAGE_PATH,
    transform=transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor()
    ])
)
dl = DataLoader(ds, BATCH_SIZE, shuffle=True)

vae_params = dict(
    image_size=IMAGE_SIZE,
    num_layers=NUM_LAYERS,
    num_tokens=NUM_TOKENS,
    codebook_dim=EMB_DIM,
    hidden_dim=HID_DIM,
    num_resnet_blocks=NUM_RESNET_BLOCKS
)
vae = DiscreteVAE(
    **vae_params,
    smooth_l1_loss=SMOOTH_L1_LOSS,
    kl_div_loss_weight=KL_LOSS_WEIGHT
).to(device)

# Check whether training data is enough
assert len(ds) > 0, 'folder does not contain any images'
logging.info(f'{len(ds)} images found for training')


def save_model(path):
    # Save the training state
    save_obj = {
        'hparams': vae_params,
        'weights': vae.state_dict(),
    }
    torch.save(save_obj, path)


# Optimizer and scheduler
opt = Adam(vae.parameters(), lr=LEARNING_RATE)
scheduler = ExponentialLR(optimizer=opt, gamma=LR_DECAY_RATE)

# Starting temperature
global_step = 0
temp = STARTING_TEMP

for epoch in range(EPOCHS):
    running = tqdm(dl, leave=False)
    for i, (images, _) in enumerate(running):
        running.set_description(f"Epoch {epoch + 1}/{EPOCHS}")
        images = images.to(device)

        loss, recons = vae(
            images,
            return_loss=True,
            return_recons=True,
            temp=temp
        )

        # Update parameters
        opt.zero_grad()
        loss.backward()
        opt.step()

        logs = {}

        if i % 100 == 0:
            k = NUM_IMAGES_SAVE

            with torch.no_grad():
                codes = vae.get_codebook_indices(images[:k])
                hard_recons = vae.decode(codes)

            images, recons = map(lambda t: t[:k], (images, recons))
            images, recons, hard_recons, codes = map(lambda t: t.detach().cpu(),
                                                     (images, recons, hard_recons, codes))
            images, recons, hard_recons = map(
                lambda t: make_grid(t.float(), nrow=int(sqrt(k)), normalize=True, range=(-1, 1)),
                (images, recons, hard_recons))

            logs = {
                **logs,
                'sample images': wandb.Image(images, caption='original images'),
                'reconstructions': wandb.Image(recons, caption='reconstructions'),
                'hard reconstructions': wandb.Image(hard_recons, caption='hard reconstructions'),
                'codebook_indices': wandb.Histogram(codes),
                'temperature': temp
            }

            save_model(f'./vae.pt')
            wandb.save('./vae.pt')

            # Temperature anneal
            temp = max(temp * math.exp(-ANNEAL_RATE * global_step), TEMP_MIN)

            # Learning rate decay
            scheduler.step()

        if i % 10 == 0:
            lr = scheduler.get_last_lr()[0]
            running.set_postfix(learning_rate=f"lr - {lr:6f}", loss=f"{loss.item()}")

            logs = {
                **logs,
                'epoch': epoch,
                'iter': i,
                'loss': loss.item(),
                'lr': lr
            }

        wandb.log(logs)
        global_step += 1

    # save trained model to wandb as an artifact every epoch's end
    model_artifact = wandb.Artifact('trained-vae',
                                    type='model',
                                    metadata=dict(model_config))
    model_artifact.add_file('vae.pt')
    run.log_artifact(model_artifact)

# Save final dVAE
save_model('./vae-final.pt')
wandb.save('./vae-final.pt')

model_artifact = wandb.Artifact('trained-vae',
                                type='model',
                                metadata=dict(model_config))
model_artifact.add_file('vae-final.pt')
run.log_artifact(model_artifact)

wandb.finish()
