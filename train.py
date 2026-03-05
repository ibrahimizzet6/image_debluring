from data_loader import DeblurDataset
from model import Discriminator, Generator
import torchvision.transforms as transforms
import torch
from torch.utils.data import DataLoader


# Hyperparameters
BATCH_SIZE = 16
LEARNING_RATE = 0.0002
EPOCHS = 300
IMAGE_SIZE = 256
NUM_WORKERS = 2
CHANNELS_IMAGE = 3
OPTIMIZER_G = torch.optim.Adam
OPTIMIZER_D = torch.optim.Adam
LOSS_FN = torch.nn.BCEWithLogitsLoss()
LOSS_FN_L1 = torch.nn.L1Loss()
LAMBDA_L1 = 100
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BETAS = (0.5, 0.999)

#transforms
transform = transforms.Compose(
    [
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ]
)



# Dataset and DataLoader
Train_Dataset = DeblurDataset(root_dir="Wider-Face/train", transform=transform)
Test_Dataset = DeblurDataset(root_dir="Wider-Face/test", transform=transform)

Train_Loader = DataLoader(Train_Dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)    
Test_Loader = DataLoader(Test_Dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

# Train Loop
generator = Generator().to(DEVICE)
discriminator = Discriminator().to(DEVICE)

optimizer_G = OPTIMIZER_G(generator.parameters(), lr=LEARNING_RATE, betas=BETAS)
optimizer_D = OPTIMIZER_D(discriminator.parameters(), lr=LEARNING_RATE, betas=BETAS)



def train(dataloader, generator, discriminator, optimizer_G, optimizer_D, loss_fn, loss_fn_l1, lambda_l1):
    generator.train()
    discriminator.train()

    for epoch in range(EPOCHS):
        for batch in dataloader:
            blur = batch["blur"].to(DEVICE)
            sharp = batch["sharp"].to(DEVICE)

            # Train Discriminator
            optimizer_D.zero_grad()

            outputs_real = discriminator(blur,sharp)
            real_labels = torch.ones_like(outputs_real).to(DEVICE)

            fake_images = generator(blur)                        
            outputs_fake = discriminator(blur,fake_images.detach())  
            fake_labels = torch.zeros_like(outputs_fake).to(DEVICE)

            loss_real = loss_fn(outputs_real, real_labels)
            loss_fake = loss_fn(outputs_fake, fake_labels)

            loss_D = (loss_real + loss_fake) / 2
            loss_D.backward()
            optimizer_D.step()

            # Train Generator
            optimizer_G.zero_grad()
            outputs_fake_for_G = discriminator(blur,fake_images)
            loss_G_adv = loss_fn(outputs_fake_for_G, real_labels)
            loss_G_l1 = loss_fn_l1(fake_images, sharp) * lambda_l1
            loss_G = loss_G_adv + loss_G_l1
            loss_G.backward()
            optimizer_G.step()

        print(f"Epoch [{epoch+1}/{EPOCHS}] Loss D: {loss_D.item():.4f}, Loss G: {loss_G.item():.4f}")


# Save Models
def save_models(generator, discriminator):
    torch.save(generator.state_dict(), "generator_epoch_pth")
    torch.save(discriminator.state_dict(), "discriminator_epoch_pth")

if __name__ == "__main__":
    train(Train_Loader, generator, discriminator, optimizer_G, optimizer_D, LOSS_FN, LOSS_FN_L1, LAMBDA_L1)
    save_models(generator, discriminator)
  