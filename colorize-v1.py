import os
import torch
from torch import nn
import torchvision
from torchvision import transforms, models as torchvision_models
from torch.utils.data import Dataset, DataLoader
import timm
import pandas as pd
from PIL import Image
from pytorch_lightning import LightningModule, Trainer, loggers, callbacks
from diffusers import StableDiffusionPipeline, AutoencoderKL

from torchvision.models import vgg16

class ColorizationDataset(Dataset):
    # data
    def __init__(self, data_folder, data_csv, transform=None):
        """
        Args:
            input_dir (string): Directory with all the input images.
            output_dir (string): Directory with all the target (color) images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.data_folder = data_folder
        self.data_path = os.path.join(data_folder, data_csv)
        self.images = pd.read_csv(self.data_path)
        self.transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=3),  # Convert grayscale to RGB by replicating channels
            transforms.ToTensor()  # Convert images to PyTorch tensors
        ])
        self.tranform_output = transforms.Compose([transforms.ToTensor()])
        
    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        sketch, colored = self.images.iloc[idx]
        sketch_image = self.transform(self.__loadImage(sketch))
        colored_image = self.tranform_output(self.__loadImage(colored))
        return sketch_image, colored_image

    def viewImage(self, idx):
        sketch, colored = self.images.iloc[idx]
        return self.__loadImage(sketch), self.__loadImage(colored)

    def __loadImage(self, image_path):
        return Image.open(os.path.join(self.data_folder, image_path))

class VGGPerceptualLoss(LightningModule):
    def __init__(self, vgg_model):
        super().__init__()
        self.vgg = vgg_model
        self.criterion = nn.MSELoss()
        self.features = list(self.vgg.features[:16])
        self.features = nn.Sequential(*self.features).eval()
        
        for params in self.features.parameters():
            params.requires_grad = False

    def forward(self, x, y):
        return self.criterion(self.features(x),self.features(y))

class Colorizer(LightningModule):
    def __init__(self, vae):
        super().__init__()
        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = vae
        vgg_model = vgg16(weights=True)
        self.loss_fn = VGGPerceptualLoss(vgg_model)

    def forward(self, x):
        return self.model(x)
    
    def configure_optimizers(self):
        self.hparams.learning_rate = 0.000005
        return torch.optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=self.hparams.learning_rate)

    def training_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self(inputs).sample
        loss = self.loss_fn(outputs, targets)
        self.log('train_loss', loss)
        return loss        

pipe = StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2")
model = Colorizer(pipe.vae)
# torch.save(model.model.state_dict(), 'vae_only.pth')

# model = Colorizer(torch.load('vae_only.pth'))
# model.model.load_state_dict(torch.load('vae_only.pth'))

data_folder = 'data/training'
data_csv = 'data.csv'
training_dataset = ColorizationDataset(data_folder, data_csv)
dataloader = DataLoader(training_dataset, batch_size=1, shuffle=True, num_workers=2)

logger = loggers.TensorBoardLogger("tb_logs")
trainer = Trainer(accelerator="gpu", devices=4, max_epochs=30, logger=logger, strategy='ddp_find_unused_parameters_true')

trainer.fit(model, dataloader)


# model.eval()
# data_folder = 'data/test'
# data_csv = 'data.csv'
# test_dataset = ColorizationDataset(data_folder, data_csv)


# model.cpu()

# idx = 3003
# x, y = test_dataset[idx]
# output = model(x.unsqueeze(0))

# input_image, output_image = test_dataset.viewImage(idx)
# input_image

# output_image

# # pipe = StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2")
# # encoder = pipe.vae.encoder
# # decoder = pipe.vae.decoder

# # x, y = training_dataset[100]
# # model.eval()
# # output = model(x.unsqueeze(0))

# viewTensor(output.sample)

# import matplotlib.pyplot as plt
# from torchvision.transforms.functional import to_pil_image

# def viewTensor(output):
#     image = to_pil_image(output.squeeze())

#     # Display the image
#     plt.imshow(image)
#     plt.axis('off')  # Turn off axis numbers and ticks
#     plt.show()


# items = list(output.items())


# # In[61]:


# _, image = items[0]


# # In[62]:


# viewTensor(image)


# # In[56]:


# image.shape


# # In[ ]:




