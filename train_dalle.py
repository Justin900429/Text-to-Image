"""Code adapted from lucidrains/DALLE-pytorch
GitHub link: https://github.com/lucidrains/DALLE-pytorch
"""
# Import built-in package
import argparse
import logging
from random import choice
from pathlib import Path

# Import torch package
import torch
from torch.optim import Adam
from torch.nn.utils import clip_grad_norm_
from PIL import Image
from torchvision import transforms as T
from torch.utils.data import DataLoader, Dataset

# Import DALLE package
from dalle_pytorch import OpenAIDiscreteVAE, VQGanVAE1024, DiscreteVAE, DALLE
from dalle_pytorch.simple_tokenizer import tokenize, tokenizer, VOCAB_SIZE

# See the training process
import wandb

# Set up logger
logging.getLogger().setLevel(logging.INFO)

# Argument parsing
parser = argparse.ArgumentParser()
group = parser.add_mutually_exclusive_group(required=False)
group.add_argument(
    '--vae_path',
    type=str,
    help='path to your trained discrete VAE'
)
group.add_argument(
    '--dalle_path',
    type=str,
    help='path to your partially trained DALL-E'
)
parser.add_argument(
    '--image_text_folder',
    type=str,
    required=True,
    help='path to your folder of images and text for learning the DALL-E'
)
parser.add_argument(
    '--taming',
    dest='taming',
    action='store_true'
)
args = parser.parse_args()


# Helpers
def exists(val):
    return val is not None


# Hyperparameters for training
VAE_PATH = args.vae_path
DALLE_PATH = args.dalle_path
RESUME = exists(DALLE_PATH)

EPOCHS = 20
BATCH_SIZE = 4
LEARNING_RATE = 3e-4
GRAD_CLIP_NORM = 0.5

# Hyperparameters for model
MODEL_DIM = 512
TEXT_SEQ_LEN = 256
DEPTH = 2
HEADS = 4
DIM_HEAD = 64
REVERSIBLE = True

# Check cuda is available or not
device = "cuda:0" if torch.cuda.is_available() else "cpu"

# Reconstitute vae
if RESUME:
    dalle_path = Path(DALLE_PATH)
    assert dalle_path.exists(), 'DALL-E model file does not exist'

    loaded_obj = torch.load(str(dalle_path))
    dalle_params, vae_params, weights = loaded_obj['hparams'], loaded_obj['vae_params'], loaded_obj['weights']

    if vae_params is not None:
        vae = DiscreteVAE(**vae_params)
    else:
        vae_klass = OpenAIDiscreteVAE if not args.taming else VQGanVAE1024
        vae = vae_klass()

    dalle_params = dict(
        **dalle_params
    )
    IMAGE_SIZE = vae.image_size
else:
    if exists(VAE_PATH):
        vae_path = Path(VAE_PATH)
        assert vae_path.exists(), 'VAE model file does not exist'

        loaded_obj = torch.load(str(vae_path))

        vae_params, weights = loaded_obj['hparams'], loaded_obj['weights']

        vae = DiscreteVAE(**vae_params)
        vae.load_state_dict(weights)
    else:
        logging.info('Using pretrained VAE for encoding images to tokens')
        vae_params = None

        vae_klass = OpenAIDiscreteVAE if not args.taming else VQGanVAE1024
        vae = vae_klass()

    IMAGE_SIZE = vae.image_size

    dalle_params = dict(
        num_text_tokens=VOCAB_SIZE,
        text_seq_len=TEXT_SEQ_LEN,
        dim=MODEL_DIM,
        depth=DEPTH,
        heads=HEADS,
        dim_head=DIM_HEAD,
        reversible=REVERSIBLE
    )


# Helpers
def save_model(path):
    # Save the training state
    save_obj = {
        'hparams': dalle_params,
        'vae_params': vae_params,
        'weights': dalle.state_dict()
    }
    torch.save(save_obj, path)


# Dataset loading
class TextImageDataset(Dataset):
    def __init__(self, folder, text_len=256, image_size=128):
        super().__init__()
        path = Path(folder)

        text_files = [*path.glob('**/*.txt')]

        image_files = [
            *path.glob('**/*.png'),
            *path.glob('**/*.jpg'),
            *path.glob('**/*.jpeg')
        ]

        text_files = {t.stem: t for t in text_files}
        image_files = {i.stem: i for i in image_files}

        keys = (image_files.keys() & text_files.keys())

        self.keys = list(keys)
        self.text_files = {k: v for k, v in text_files.items() if k in keys}
        self.image_files = {k: v for k, v in image_files.items() if k in keys}

        self.image_transform = T.Compose([
            T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
            T.RandomResizedCrop(image_size, scale=(0.6, 1.), ratio=(1, 1)),
            T.ToTensor()
        ])

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, ind):
        key = self.keys[ind]
        text_file = self.text_files[key]
        image_file = self.image_files[key]

        cur_image = Image.open(image_file)
        descriptions = text_file.read_text().split('\n')
        descriptions = list(filter(lambda t: len(t) > 0, descriptions))
        description = choice(descriptions)

        tokenized_text = tokenize(description).squeeze(0)
        cur_mask = tokenized_text != 0

        image_tensor = self.image_transform(cur_image)
        return tokenized_text, image_tensor, cur_mask


# Create dataset and dataloader
ds = TextImageDataset(
    args.image_text_folder,
    text_len=TEXT_SEQ_LEN,
    image_size=IMAGE_SIZE
)

# Check whether users have data to train
assert len(ds) > 0, 'dataset is empty'
logging.info(f'{len(ds)} image-text pairs found for training')

dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

# Initialize DALL-E
dalle = DALLE(vae=vae, **dalle_params).cuda()

if RESUME:
    dalle.load_state_dict(weights)

# Optimizer
opt = Adam(dalle.parameters(), lr=LEARNING_RATE)

model_config = dict(
    depth=DEPTH,
    heads=HEADS,
    dim_head=DIM_HEAD
)

run = wandb.init(project='dalle_train_transformer',
                 resume=RESUME,
                 config=model_config)

# Training
for epoch in range(EPOCHS):
    for i, (text, images, mask) in enumerate(dl):
        text, images, mask = map(lambda t: t.to(device), (text, images, mask))

        loss = dalle(text, images, mask=mask, return_loss=True)

        # Update parameters
        loss.backward()
        clip_grad_norm_(dalle.parameters(), GRAD_CLIP_NORM)

        opt.step()
        opt.zero_grad()

        log = {}

        if i % 10 == 0:
            logging.info(f"{epoch}, {i}, loss - {loss.item()}")

            log = {
                **log,
                'epoch': epoch,
                'iter': i,
                'loss': loss.item()
            }

        if i % 100 == 0:
            sample_text = text[:1]
            token_list = sample_text.masked_select(sample_text != 0).tolist()
            decoded_text = tokenizer.decode(token_list)

            image = dalle.generate_images(
                text[:1],
                mask=mask[:1],
                filter_thres=0.9  # topk sampling at 0.9
            )

            save_model(f'./dalle.pt')
            wandb.save(f'./dalle.pt')

            log = {
                **log,
                'image': wandb.Image(image, caption=decoded_text)
            }

        wandb.log(log)

    # save trained model to wandb as an artifact every epoch's end
    model_artifact = wandb.Artifact('trained-dalle',
                                    type='model',
                                    metadata=dict(model_config))
    model_artifact.add_file('dalle.pt')
    run.log_artifact(model_artifact)

# Save final DALLE
save_model(f'./dalle-final.pt')
wandb.save('./dalle-final.pt')
model_artifact = wandb.Artifact('trained-dalle',
                                type='model',
                                metadata=dict(model_config))
model_artifact.add_file('dalle-final.pt')
run.log_artifact(model_artifact)

wandb.finish()
