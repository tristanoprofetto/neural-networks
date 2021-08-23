import torch
from torch import nn
from tqdm.auto import tqdm
from torchvision import transforms
from torchvision.datasets import MNIST
from torchvision.utils import make_grid
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

torch.manual_seed(0)

# Function to display images
def show_images(images, n, size=(1, 28, 28)):
    images = (images + 1) / 2
    image_unflat = images.detach().cpu()
    image_grid = make_grid(image_unflat[:n], nrow=5)
    plt.imshow(image_grid.permute(1, 2, 0).squeeze())
    plt.show()

# Function for generating noise vectors
def getNoise(n, noise_dim, device='cpu'):
    return torch.randn(n, noise_dim, device=device)

# Defining the Generator component of the DCGAN
class Generator(nn.Module):

    def __init__(self, noise_dim=10, n_channel=1, hidden_dim=64):
        super(Generator, self).__init__()
        self.noise_dim = noise_dim

        self.gen = nn.Sequential(
            self.gen_block(noise_dim, hidden_dim * 4),
            self.gen_block(hidden_dim * 4, hidden_dim * 2, kernel_size=4, stride=1),
            self.gen_block(hidden_dim * 2, hidden_dim),
            self.gen_block(hidden_dim, n_channel, kernel_size=4, final_layer=True)
        )


    def gen_block(self, input_channels, output_channels, kernel_size=3, stride=2, final_layer=False):

        # Neural Block
        if not final_layer:
            return nn.Sequential(
                nn.ConvTranspose2d(input_channels, output_channels, kernel_size=kernel_size, stride=stride),
                nn.BatchNorm2d(output_channels),
                nn.ReLU(inplace=True)
            )
        # Final Layer
        else:
            return nn.Sequential(
                nn.ConvTranspose2d(input_channels, output_channels, kernel_size, stride),
                nn.Tanh()
            )

    # Function for returning the noise tensor with adjusted dimensions
    def unsqueezeNoise(self, noise):
        return noise.view(len(noise), self.noise_dim, 1, 1)

    # Function for returning the generated images
    def forward(self, noise):

        x= self.unsqueezeNoise(noise)
        return self.gen(x)


# Defining the Discriminator component of the DCGAN
class Discriminator(nn.Module):

    def __init__(self, n_channels, hidden_dim=16):
        super(Discriminator, self).__init__()
        self.disc = nn.Sequential(
            self.disc_block(n_channels, hidden_dim),
            self.disc_block(hidden_dim, hidden_dim * 2),
            self.disc_gen(hidden_dim * 2, 1, final_layer=True)
        )

    def disc_block(self, input_channels, output_channels, kernel_size=4, stride=2, final_layer=False):

        if not final_layer:
            return nn.Sequential(
                nn.Conv2d(input_channels, output_channels, kernel_size, stride),
                nn.BatchNorm2d(output_channels),
                nn.LeakyReLU(0.2, inplace=True)
            )

        else:
            return nn.Sequential(
                nn.Conv2d(input_channels, output_channels, kernel_size, stride)
            )


    def forward(self, image):
        disc_prediction = self.disc(image)
        return disc_prediction.view(len(disc_prediction), -1)



# Model Training
criterion = nn.BCEWithLogitsLoss()
z_dim = 64
display_step = 500
batch_size = 128
# A learning rate of 0.0002 works well on DCGAN
lr = 0.0002

# These parameters control the optimizer's momentum, which you can read more about here:
# https://distill.pub/2017/momentum/ but you don’t need to worry about it for this course!
beta_1 = 0.5
beta_2 = 0.999
device = 'cuda'

# You can tranform the image values to be between -1 and 1 (the range of the tanh activation)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
])

dataloader = DataLoader(
    MNIST('.', download=True, transform=transform),
    batch_size=batch_size,
    shuffle=True)

gen = Generator(z_dim).to(device)
gen_opt = torch.optim.Adam(gen.parameters(), lr=lr, betas=(beta_1, beta_2))
disc = Discriminator().to(device)
disc_opt = torch.optim.Adam(disc.parameters(), lr=lr, betas=(beta_1, beta_2))

# You initialize the weights to the normal distribution
# with mean 0 and standard deviation 0.02
def weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
    if isinstance(m, nn.BatchNorm2d):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
        torch.nn.init.constant_(m.bias, 0)
gen = gen.apply(weights_init)
disc = disc.apply(weights_init)

epochs = 10
cur_steep = 0
mean_generator_loss = 0
mean_discriminator_loss = 0

for epoch in range(epochs):
    for real, _ in tqdm(dataloader):
        batch_size = len(real)
        real = real.to(device)

        # Updating the Discriminator
        disc_opt.zero_grad()
        fake_noise = getNoise(batch_size, z_dim, device)
        fake = gen(fake_noise)

        disc_fake = disc(fake.detach())
        disc_fake_loss = criterion(disc_fake, torch.zeros_like(disc_fake))

        disc_real = disc(real)
        disc_real_loss = criterion(disc_real, torch.ones_like(disc_real))

        disc_loss = (disc_fake_loss + disc_real_loss) / 2


        # Updating Generator
        gen_opt.zero_grad()
        fake_noise_gen = get_noise(batch_size, z_dim)
        fake_2 = gen(fake_noise_gen)
        disc_fake = disc(fake_2)
        gen_loss = criterion(disc_fake, torch.ones_like(disc_fake))
        gen_loss.backward()
        gen_opt.step()


        # Average Discriminator Loss
        mean_discriminator_loss += disc_loss.item() / display_step
        disc_loss.backward(retain_graph=True)
        disc_opt.step()

        # Average Generator Loss
        mean_generator_loss += gen_loss.item() / display_step


        # Visualize training process
        if cur_steep % display_step == 0 and cur_steep > 0:
            print(f"Step {cur_steep}: \nGenerator Loss: {mean_generator_loss}, \nDiscriminator Loss: {mean_discriminator_loss}")
            show_images(fake)
            show_images(real)
            mean_generator_loss = 0
            mean_discriminator_loss =0
        cur_steep += 1

