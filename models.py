import os
import random
import time
from typing import Tuple
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import tqdm
from .theme import SUCCESS_STYLE, console, INFO_STYLE


class Generator(nn.Module):
    """
    The generator, GG, is designed to map the latent space vector (zz) to data-space.
    Since our data are images, converting zz to data-space means ultimately creating a RGB image with the same size as the training images (i.e. 3x64x64).
    In practice, this is accomplished through a series of strided two dimensional convolutional transpose layers,
    each paired with a 2d batch norm layer and a relu activation.
    The output of the generator is fed through a tanh function to return it to the input data range of [−1,1][−1,1].
    It is worth noting the existence of the batch norm functions after the conv-transpose layers, as this is a critical contribution of the DCGAN paper.
    These layers help with the flow of gradients during training.

    Attributes:
        ngpu (int): Number of GPUs available.
        main (torch.nn.Sequential): The sequential model.
    """

    def __init__(self, ngpu: int, nz: int, ngf: int, nc: int):
        """
        Initializes the Generator.

        Args:
            ngpu (int): Number of GPUs available.
            nz (int): Size of the latent z vector.
            ngf (int): Size of feature maps in the generator.
            nc (int): Number of channels in the output images.
        """
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. ``(ngf*8) x 4 x 4``
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. ``(ngf*4) x 8 x 8``
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. ``(ngf*2) x 16 x 16``
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. ``(ngf) x 32 x 32``
            nn.ConvTranspose2d(ngf, ngf // 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf // 2),
            nn.ReLU(True),
            # state size. ``(ngf//2) x 64 x 64``
            nn.ConvTranspose2d(ngf // 2, nc, 4, 2, 1, bias=False),
            nn.Tanh(),
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the generator.

        Args:
            input (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        return self.main(input)


class Discriminator(nn.Module):
    """
    The discriminator is a binary classification network that takes an image as input and outputs a scalar probability that the input image is real (as opposed to fake)
    D takes a input image, processes it through a series of Conv2d, BatchNorm2d, and LeakyReLU layers,
    and outputs the final probability through a Sigmoid activation function.
    This architecture can be extended with more layers if necessary for the problem, but there is significance to the use of the strided convolution, BatchNorm, and LeakyReLUs.
    The DCGAN paper mentions it is a good practice to use strided convolution rather than pooling to downsample because it lets the network learn its own pooling function.
    Also batch norm and leaky relu functions promote healthy gradient flow which is critical for the learning process of both GG and DD.

    Attributes:
        ngpu (int): Number of GPUs available.
        main (torch.nn.Sequential): The sequential model.
    """

    def __init__(self, ngpu: int, nc: int, ndf: int):
        """
        Initializes the Discriminator.

        Args:
            ngpu (int): Number of GPUs available.
            nc (int): Number of channels in the input images.
            ndf (int): Size of feature maps in the discriminator.
        """
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is ``(nc) x 128 x 128``
            nn.Conv2d(nc, ndf // 2, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(ndf//2) x 64 x 64``
            nn.Conv2d(ndf // 2, ndf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(ndf) x 32 x 32``
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(ndf*2) x 16 x 16``
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(ndf*4) x 8 x 8``
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(ndf*8) x 4 x 4``
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the discriminator.

        Args:
            input (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        return self.main(input)


class DCGAN:
    """
    A Deep Convolutional Generative Adversarial Network (DCGAN).

    Attributes:
        ngpu (int): Number of GPUs available.
        nz (int): Size of the latent z vector.
        ngf (int): Size of feature maps in the generator.
        ndf (int): Size of feature maps in the discriminator.
        nc (int): Number of channels in the images.
        lr (float): Learning rate.
        beta1 (float): Beta1 hyperparameter for Adam optimizers.
        dataroot (str): Root directory for the dataset.
        batch_size (int): Batch size during training.
        image_size (int): Spatial size of training images.
        workers (int): Number of worker threads for loading the data.
        num_epochs (int): Number of training epochs.
        device (torch.device): Device on which to run the model.
    """

    def __init__(
        self,
        dataroot: str,
        ngpu: int = 1,
        nz: int = 100,
        ngf: int = 64,
        ndf: int = 64,
        nc: int = 3,
        lr: float = 0.0002,
        beta1: float = 0.5,
        batch_size: int = 128,
        workers: int = 2,
        num_epochs: int = 5,
    ):
        self.ngpu = ngpu
        self.nz = nz
        self.ngf = ngf
        self.ndf = ndf
        self.nc = nc
        self.lr = lr
        self.beta1 = beta1
        self.dataroot = dataroot
        self.batch_size = batch_size
        self.image_size = 128
        self.workers = workers
        self.num_epochs = num_epochs
        self.device = torch.device(
            "cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu"
        )

        self.dataloader = self.load_data()
        self.netG, self.netD = self.initialize_models()
        self.criterion, self.optimizerD, self.optimizerG = self.setup_optimizers()

        # Create batch of latent vectors for visualization
        self.fixed_noise = torch.randn(self.ngf, self.nz, 1, 1, device=self.device)

        # Establish convention for real and fake labels during training
        self.real_label = 1.0
        self.fake_label = 0.0

    def load_data(self) -> torch.utils.data.DataLoader:
        """
        Loads the dataset.

        Returns:
            torch.utils.data.DataLoader: The data loader for the dataset.
        """
        dataset = dset.ImageFolder(
            root=self.dataroot,
            transform=transforms.Compose(
                [
                    transforms.Resize(self.image_size),
                    transforms.CenterCrop(self.image_size),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ]
            ),
        )
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.workers
        )
        return dataloader

    def weights_init(self, m: nn.Module):
        """
        From the DCGAN paper, the authors specify that all model weights shall be randomly initialized from a Normal distribution with mean=0, stdev=0.02.
        The weights_init function takes an initialized model as input and reinitializes all convolutional, convolutional-transpose, and batch normalization layers to meet this criteria.
        This function is applied to the models immediately after initialization.
        Args:
            m (nn.Module): The model to initialize.
        """
        classname = m.__class__.__name__
        if classname.find("Conv") != -1:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find("BatchNorm") != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)

    def initialize_models(self) -> Tuple[nn.Module, nn.Module]:
        """
        Initializes the Generator and Discriminator models.

        Returns:
            Tuple[nn.Module, nn.Module]: The initialized Generator and Discriminator models.
        """
        netG = Generator(self.ngpu, self.nz, self.ngf, self.nc).to(self.device)
        netG.apply(self.weights_init)

        netD = Discriminator(self.ngpu, self.nc, self.ndf).to(self.device)
        netD.apply(self.weights_init)

        return netG, netD

    def setup_optimizers(self) -> Tuple[nn.BCELoss, optim.Adam, optim.Adam]:
        """
        Sets up the optimizers for the Generator and Discriminator.

        Returns:
            Tuple[nn.BCELoss, optim.Adam, optim.Adam]: The loss function and optimizers.
        """
        criterion = nn.BCELoss()

        optimizerD = optim.Adam(
            self.netD.parameters(), lr=self.lr, betas=(self.beta1, 0.999)
        )
        optimizerG = optim.Adam(
            self.netG.parameters(), lr=self.lr, betas=(self.beta1, 0.999)
        )

        return criterion, optimizerD, optimizerG

    def save_models(
        self, checkpoint_dir: str = "pythopix_results/train/dcgan/checkpoints"
    ):
        """
        Saves the entire trained Generator and Discriminator models.

        Args:
            checkpoint_dir (str): Directory to save the model checkpoints.
        """
        os.makedirs(checkpoint_dir, exist_ok=True)
        torch.save(self.netG, os.path.join(checkpoint_dir, "generator.pt"))
        torch.save(self.netD, os.path.join(checkpoint_dir, "discriminator.pt"))

    def train(
        self,
        checkpoint_interval: int = 5,
        best_model_criteria: str = "lossG",
        seed: int = 999,
    ):
        """
        Here, we will closely follow Algorithm 1 from the Goodfellow's paper, while abiding by some of the best practices shown in ganhacks.
        Namely, we will “construct different mini-batches for real and fake” images,
        and also adjust G's objective function to maximize log(D(G(z))).
        Training is split up into two main parts. Part 1 updates the Discriminator and Part 2 updates the Generator.

        The goal of training the discriminator is to maximize the probability of correctly classifying a given input as real or fake.
        In terms of Goodfellow, we wish to “update the discriminator by ascending its stochastic gradient”.
        Practically, we want to maximize log(D(x))+log(1-D(G(z))).
        Due to the separate mini-batch suggestion from ganhacks, we will calculate this in two steps.
        First, we will construct a batch of real samples from the training set, forward pass through D, calculate the loss log(D(x) then calculate the gradients in a backward pass.
        Secondly, we will construct a batch of fake samples with the current generator, forward pass this batch through D, calculate the loss log(1-D(G(z)),
        and accumulate the gradients with a backward pass. Now, with the gradients accumulated from both the all-real and all-fake batches,
        we call a step of the Discriminator's optimizer.

        We want to train the Generator by minimizing log(1-D(G(z))) in an effort to generate better fakes.
        As mentioned, this was shown by Goodfellow to not provide sufficient gradients, especially early in the learning process.
        As a fix, we instead wish to maximize log(D(G(z))).
        In the code we accomplish this by: classifying the Generator output from Part 1 with the Discriminator,
        computing G's loss using real labels as GT, computing G's gradients in a backward pass, and finally updating G's parameters with an optimizer step.
        It may seem counter-intuitive to use the real labels as GT labels for the loss function, but this allows us to use the log(x) part of the BCELoss (rather than the log(1-x))
        which is exactly what we want.

        Args:
            checkpoint_interval (int): Interval (in epochs) at which to save a checkpoint of the model.
            best_model_criteria (str): Criteria to decide the best model ('lossD' or 'lossG').
            seed (int): Random seed for reproducibility.

        Returns:
            G_losses (list): A list of losses for the Generator recorded during training.
            D_losses (list): A list of losses for the Discriminator recorded during training.
            img_list (list): A list of generated image grids, each grid is saved after a certain number of iterations.
        """
        random.seed(seed)
        torch.manual_seed(seed)
        torch.use_deterministic_algorithms(True)
        start_time = time.time()

        img_list = []
        G_losses = []
        D_losses = []
        iters = 0
        best_lossD = float("inf")
        best_lossG = float("inf")

        os.makedirs("pythopix_results/train/dcgan/figs", exist_ok=True)
        real_batch = next(iter(self.dataloader))
        plt.figure(figsize=(8, 8))
        plt.axis("off")
        plt.title("Training Images")
        plt.imshow(
            np.transpose(
                vutils.make_grid(
                    real_batch[0].to(self.device)[:64], padding=2, normalize=True
                ).cpu(),
                (1, 2, 0),
            )
        )
        plt.savefig("pythopix_results/train/dcgan/figs/training_images.png")

        console.print("Starting Training Loop...", style=INFO_STYLE)

        # For each epoch
        for epoch in range(self.num_epochs):
            # For each batch in the dataloader
            for i, data in enumerate(self.dataloader, 0):
                ############################
                # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
                ###########################
                ## Train with all-real batch
                self.netD.zero_grad()
                # Format batch
                real_cpu = data[0].to(self.device)
                b_size = real_cpu.size(0)
                label = torch.full(
                    (b_size,), self.real_label, dtype=torch.float, device=self.device
                )
                # Forward pass real batch through D
                output = self.netD(real_cpu).view(-1)
                # Calculate loss on all-real batch
                errD_real = self.criterion(output, label)
                # Calculate gradients for D in backward pass
                errD_real.backward()
                D_x = output.mean().item()

                ## Train with all-fake batch
                # Generate batch of latent vectors
                noise = torch.randn(b_size, self.nz, 1, 1, device=self.device)
                # Generate fake image batch with G
                fake = self.netG(noise)
                label.fill_(self.fake_label)
                # Classify all fake batch with D
                output = self.netD(fake.detach()).view(-1)
                # Calculate D's loss on the all-fake batch
                errD_fake = self.criterion(output, label)
                # Calculate the gradients for this batch, accumulated (summed) with previous gradients
                errD_fake.backward()
                D_G_z1 = output.mean().item()
                # Compute error of D as sum over the fake and the real batches
                errD = errD_real + errD_fake
                # Update D
                self.optimizerD.step()

                ############################
                # (2) Update G network: maximize log(D(G(z)))
                ###########################
                self.netG.zero_grad()
                label.fill_(self.real_label)  # fake labels are real for generator cost
                # Since we just updated D, perform another forward pass of all-fake batch through D
                output = self.netD(fake).view(-1)
                # Calculate G's loss based on this output
                errG = self.criterion(output, label)
                # Calculate gradients for G
                errG.backward()
                D_G_z2 = output.mean().item()
                # Update G
                self.optimizerG.step()

                # Output training stats
                if i % 50 == 0:
                    print(
                        "[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f"
                        % (
                            epoch,
                            self.num_epochs,
                            i,
                            len(self.dataloader),
                            errD.item(),
                            errG.item(),
                            D_x,
                            D_G_z1,
                            D_G_z2,
                        )
                    )

                # Save Losses for plotting later
                G_losses.append(errG.item())
                D_losses.append(errD.item())

                # Check how the generator is doing by saving G's output on fixed_noise
                if (iters % 500 == 0) or (
                    (epoch == self.num_epochs - 1) and (i == len(self.dataloader) - 1)
                ):
                    with torch.no_grad():
                        fake = self.netG(self.fixed_noise).detach().cpu()
                        img_grid = vutils.make_grid(fake, padding=2, normalize=True)

                    img_list.append(img_grid)

                    save_path = os.path.join(
                        "pythopix_results/train/dcgan", f"generator_iter{iters}.png"
                    )
                    vutils.save_image(img_grid, save_path)

                iters += 1

            if best_model_criteria == "lossD" and errD.item() < best_lossD:
                best_lossD = errD.item()
                self.save_models(checkpoint_dir="pythopix_results/train/dcgan/best")
            elif best_model_criteria == "lossG" and errG.item() < best_lossG:
                best_lossG = errG.item()
                self.save_models(checkpoint_dir="pythopix_results/train/dcgan/best")

            # Save checkpoint every few epochs
            if epoch % checkpoint_interval == 0 or epoch == self.num_epochs - 1:
                self.save_models(
                    checkpoint_dir=f"pythopix_results/train/dcgan/checkpoints/epoch_{epoch}"
                )
        end_time = time.time()
        console.print(
            f"Successfuly trained DCGAN, took {round(end_time-start_time,2)}s",
            style=SUCCESS_STYLE,
        )

        # Plot and save training losses
        plt.figure(figsize=(10, 5))
        plt.title("Generator and Discriminator Loss During Training")
        plt.plot(G_losses, label="Generator")
        plt.plot(D_losses, label="Discriminator")
        plt.xlabel("Iterations")
        plt.ylabel("Loss")
        plt.legend()
        loss_plot_path = "pythopix_results/train/dcgan/training_loss_plot.png"
        plt.savefig(loss_plot_path)
        plt.close()

        return G_losses, D_losses, img_list
