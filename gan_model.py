import torch
import torch.nn as nn
import torch.optim as optim
import torchvision # Оставляем этот импорт
import torchvision.transforms as transforms
from torchvision.utils import save_image, make_grid
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import os
import glob

# --- Гиперпараметры и Конфигурация ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}") # Эта строка должна быть одна в начале

# Скорости обучения (Learning Rates)
LEARNING_RATE_G = 2e-4  # Скорость обучения для Генератора (например, 0.0002) Обычно 2e-4
LEARNING_RATE_D = 1e-5  # Скорость обучения для Дискриминатора (например, 0.0001 - медленнее) Обычно 2e-4

BATCH_SIZE = 64      # Размер пакета данных (подбери под свою VRAM, можно начать с 32 или 64)
IMAGE_SIZE = 64      # Размер изображений (например, 64x64)
CHANNELS_IMG = 3     # Каналы изображения (3 для RGB, 1 для grayscale)
NOISE_DIM = 100      # Размерность вектора шума для генератора
NUM_EPOCHS = 10000    # Количество эпох обучения (подбирай сам)

# Параметры для балансировки обучения
SMOOTH_REAL_LABEL = 0.8 # Метка для реальных изображений (вместо 1.0) с 0.9 до 0.8 поменял
DISC_UPDATE_INTERVAL = 4 # Обновлять Дискриминатор каждый N-й шаг Генератора (например, раз в 2 шага) БЫЛО 2 ПОСТАВИЛ 4 ЧТОБ ЗАМЕДЛИТЬ ДИСКРИМИНАТОР ВЕРНУТЬ ПОТОМ

# Пути
DATASET_PATH = "dataset/mydata" # УКАЖИ ПУТЬ К ПАПКЕ С ИЗОБРАЖЕНИЯМИ в dataset
OUTPUT_PATH = "output_images"
CHECKPOINT_SAVE_INTERVAL = 20 # Сохранять чекпоинт каждые N эпох

# --- Класс для своего датасета ---
class CustomImageDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        # Находим все файлы jpg, png, jpeg (можно добавить и другие форматы)
        self.image_paths = glob.glob(os.path.join(image_dir, "*.jpg")) + \
                           glob.glob(os.path.join(image_dir, "*.png")) + \
                           glob.glob(os.path.join(image_dir, "*.jpeg"))
        if not self.image_paths:
            raise Exception(f"No images found in {image_dir}. Check the DATASET_PATH and image formats.")
        print(f"Found {len(self.image_paths)} images in {image_dir}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        try:
            image = Image.open(img_path).convert("RGB") # Конвертируем в RGB
        except Exception as e:
            print(f"Warning: Could not load image {img_path}: {e}. Trying next.")
            # Простой способ обработки ошибки: пытаемся загрузить следующий валидный элемент
            # Это не идеальный способ, но для простоты...
            # В более сложных сценариях можно логировать ошибки или возвращать заполнитель
            next_idx = (idx + 1) % len(self.image_paths) # Переходим к следующему, зацикливаясь
            if next_idx == idx: # Если в датасете только одно битое изображение
                 raise Exception(f"Single image in dataset is corrupted: {img_path}")
            return self.__getitem__(next_idx)


        if self.transform:
            image = self.transform(image)
        return image

# --- Генератор ---
class Generator(nn.Module):
    def __init__(self, noise_dim, channels_img, features_g):
        super(Generator, self).__init__()
        self.net = nn.Sequential(
            # Вход: N x noise_dim x 1 x 1
            self._block(noise_dim, features_g * 16, 4, 1, 0),  # N x f_g*16 x 4 x 4
            self._block(features_g * 16, features_g * 8, 4, 2, 1), # N x f_g*8 x 8 x 8
            self._block(features_g * 8, features_g * 4, 4, 2, 1), # N x f_g*4 x 16 x 16
            self._block(features_g * 4, features_g * 2, 4, 2, 1), # N x f_g*2 x 32 x 32
            nn.ConvTranspose2d(
                features_g * 2, channels_img, kernel_size=4, stride=2, padding=1
            ), # N x channels_img x 64 x 64
            nn.Tanh() # Нормализует выход в [-1, 1]
        )

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose2d(
                in_channels, out_channels, kernel_size, stride, padding, bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
        )

    def forward(self, x):
        return self.net(x)

# --- Дискриминатор ---
class Discriminator(nn.Module):
    def __init__(self, channels_img, features_d):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            # Вход: N x channels_img x 64 x 64
            nn.Conv2d(channels_img, features_d, kernel_size=4, stride=2, padding=1), # N x f_d x 32 x 32
            nn.LeakyReLU(0.2, inplace=True),
            self._block(features_d, features_d * 2, 4, 2, 1),    # N x f_d*2 x 16 x 16
            self._block(features_d * 2, features_d * 4, 4, 2, 1),  # N x f_d*4 x 8 x 8
            self._block(features_d * 4, features_d * 8, 4, 2, 1),  # N x f_d*8 x 4 x 4
            nn.Conv2d(features_d * 8, 1, kernel_size=4, stride=1, padding=0), # N x 1 x 1 x 1
            nn.Sigmoid() # Выход: вероятность (одно число)
        )

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels, kernel_size, stride, padding, bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        return self.net(x).view(-1) # Убираем лишние размерности

# --- Инициализация весов (хорошая практика) ---
def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)
            if m.bias is not None: # BatchNorm не имеет bias если affine=False, ConvTranspose2d может иметь bias=False
                 nn.init.constant_(m.bias.data, 0)


# --- Главный блок ---
if __name__ == "__main__":
    # Создаем директории, если их нет
    if not os.path.exists(OUTPUT_PATH):
        os.makedirs(OUTPUT_PATH)
    if not os.path.exists(DATASET_PATH):
        print(f"Error: Dataset path {DATASET_PATH} does not exist. Please create it and put your images there.")
        exit()
    elif not os.path.exists(os.path.join(DATASET_PATH)) or not any(fname.lower().endswith(('.png', '.jpg', '.jpeg')) for fname in os.listdir(DATASET_PATH)):
        print(f"Error: Dataset path {DATASET_PATH} is empty or contains no images. Please put your images there.")
        exit()


    # Трансформации для изображений
    transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(), # Преобразует PIL Image (H x W x C) в тензор (C x H x W) и нормализует в [0, 1]
        transforms.Normalize(
            [0.5 for _ in range(CHANNELS_IMG)], # Среднее
            [0.5 for _ in range(CHANNELS_IMG)]  # Стандартное отклонение
        ) # Нормализует в [-1, 1]
    ])
    
    try:
        dataset = CustomImageDataset(image_dir=DATASET_PATH, transform=transform)
        if len(dataset) == 0: # Дополнительная проверка, если CustomImageDataset не вызвал исключение
            print(f"No images were loaded from {DATASET_PATH}. Ensure images are present and paths are correct.")
            exit()
    except Exception as e:
        print(f"Error loading dataset: {e}")
        exit()

    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True, drop_last=True) # drop_last=True может помочь, если последний батч слишком мал

    # --- Инициализация или Загрузка Моделей и Оптимизаторов ---
    START_EPOCH = 0
    CHECKPOINT_LOAD_PATH = f"{OUTPUT_PATH}/checkpoint_latest.pth" 

    gen = Generator(NOISE_DIM, CHANNELS_IMG, features_g=64).to(DEVICE)
    disc = Discriminator(CHANNELS_IMG, features_d=64).to(DEVICE)
    
    opt_gen = optim.Adam(gen.parameters(), lr=LEARNING_RATE_G, betas=(0.5, 0.999))
    opt_disc = optim.Adam(disc.parameters(), lr=LEARNING_RATE_D, betas=(0.5, 0.999)) # Используем LEARNING_RATE_D

    # Попытка загрузить чекпоинт
    if os.path.exists(CHECKPOINT_LOAD_PATH):
        print(f"🔄 Loading checkpoint from {CHECKPOINT_LOAD_PATH}")
        try:
            checkpoint = torch.load(CHECKPOINT_LOAD_PATH, map_location=DEVICE)
            
            gen.load_state_dict(checkpoint['gen_state_dict'])
            disc.load_state_dict(checkpoint['disc_state_dict'])
            opt_gen.load_state_dict(checkpoint['opt_gen_state_dict'])
            opt_disc.load_state_dict(checkpoint['opt_disc_state_dict'])
            START_EPOCH = checkpoint['epoch']
            fixed_noise = checkpoint['fixed_noise'].to(DEVICE) 
            
            print(f"✅ Checkpoint loaded. Resuming training from epoch {START_EPOCH}")
        except Exception as e:
            print(f"Error loading checkpoint: {e}. Starting from scratch.")
            START_EPOCH = 0 # Сбрасываем на случай ошибки загрузки
            initialize_weights(gen)
            initialize_weights(disc)
            fixed_noise = torch.randn(32, NOISE_DIM, 1, 1).to(DEVICE) # 32 примера для отслеживания
    else:
        print("ℹ️ No checkpoint found. Starting training from scratch.")
        initialize_weights(gen)
        initialize_weights(disc)
        fixed_noise = torch.randn(32, NOISE_DIM, 1, 1).to(DEVICE)


    criterion = nn.BCELoss() # Binary Cross Entropy

    print("🚀 Starting Training...")
    # Общий счетчик шагов для DISC_UPDATE_INTERVAL, если batch_idx сбрасывается каждый батч
    # Но batch_idx в DataLoader уже работает как надо для этого внутри эпохи.
    
    for epoch in range(START_EPOCH, NUM_EPOCHS):
        for batch_idx, real_imgs in enumerate(dataloader):
            real_imgs = real_imgs.to(DEVICE)
            current_batch_size = real_imgs.shape[0] 

            if current_batch_size == 0: 
                continue

            # --- Обучение Генератора ---
            # Генератор обновляется на каждом шаге
            gen.zero_grad()
            noise_g = torch.randn(current_batch_size, NOISE_DIM, 1, 1).to(DEVICE) # Шум для генератора
            fake_imgs_for_gen = gen(noise_g)
            # Генератор хочет, чтобы дискриминатор думал, что это настоящие (метка 1.0 или SMOOTH_REAL_LABEL)
            # Использование 1.0 для генератора обычно стандартно, даже если для дискриминатора используется сглаживание
            label_real_for_gen = torch.full((current_batch_size,), 1.0, dtype=torch.float, device=DEVICE) 
            output_fake_for_gen = disc(fake_imgs_for_gen)
            loss_gen = criterion(output_fake_for_gen, label_real_for_gen) 
            loss_gen.backward()
            opt_gen.step()

            # --- Обучение Дискриминатора ---
            # Дискриминатор обновляется реже (каждый DISC_UPDATE_INTERVAL шаг)
            if (batch_idx + 1) % DISC_UPDATE_INTERVAL == 0: # +1 чтобы первый батч (batch_idx=0) тоже мог обновиться если интервал=1
                disc.zero_grad()

                # Настоящие изображения (с использованием SMOOTH_REAL_LABEL)
                label_real = torch.full((current_batch_size,), SMOOTH_REAL_LABEL, dtype=torch.float, device=DEVICE)
                output_real = disc(real_imgs)
                loss_disc_real = criterion(output_real, label_real)

                # Сгенерированные изображения (генератор уже сделал шаг, так что fake_imgs_for_gen свежие)
                # Используем .detach() для поддельных изображений при обучении дискриминатора
                noise_d = torch.randn(current_batch_size, NOISE_DIM, 1, 1).to(DEVICE) # Можно использовать тот же шум, что и для G, но лучше новый
                # fake_imgs_for_disc = gen(noise_d).detach() # Генерируем новые и отсоединяем
                # Или используем те, что уже сгенерированы для G, но отсоединяем
                fake_imgs_for_disc = fake_imgs_for_gen.detach()


                label_fake = torch.full((current_batch_size,), 0.0, dtype=torch.float, device=DEVICE) # Метки "0" для поддельных
                output_fake_disc = disc(fake_imgs_for_disc)
                loss_disc_fake = criterion(output_fake_disc, label_fake)

                loss_disc = (loss_disc_real + loss_disc_fake) / 2
                loss_disc.backward()
                opt_disc.step()
            else:
                # Если не обновляем дискриминатор, можно просто пропустить или залогировать его текущие потери без backward
                # Для простоты, просто пропускаем backward и step
                # Можно вычислить loss_disc для логирования, если нужно, но без backward()
                with torch.no_grad():
                    output_real_no_grad = disc(real_imgs)
                    loss_disc_real_no_grad = criterion(output_real_no_grad, torch.full((current_batch_size,), SMOOTH_REAL_LABEL, dtype=torch.float, device=DEVICE))
                    
                    # fake_imgs_for_disc_no_grad = gen(torch.randn(current_batch_size, NOISE_DIM, 1, 1).to(DEVICE)).detach()
                    fake_imgs_for_disc_no_grad = fake_imgs_for_gen.detach() # Используем те же, что и для G

                    output_fake_disc_no_grad = disc(fake_imgs_for_disc_no_grad)
                    loss_disc_fake_no_grad = criterion(output_fake_disc_no_grad, torch.full((current_batch_size,), 0.0, dtype=torch.float, device=DEVICE))
                    loss_disc = (loss_disc_real_no_grad + loss_disc_fake_no_grad) / 2 # Это только для логирования

            # --- Вывод информации и сохранение изображений ---
            if batch_idx % 100 == 0: # Логируем каждые 100 батчей
                print(
                    f"Epoch [{epoch+1}/{NUM_EPOCHS}] Batch {batch_idx}/{len(dataloader)} \t"
                    f"Loss D: {loss_disc.item():.4f}, Loss G: {loss_gen.item():.4f}"
                )

                with torch.no_grad(): 
                    fake_samples = gen(fixed_noise)
                    img_grid_fake = make_grid(fake_samples, normalize=True)
                    save_image(img_grid_fake, f"{OUTPUT_PATH}/sample_epoch_{epoch+1}_batch_{batch_idx}.png")

        # Сохранение чекпоинта (в конце каждой эпохи или каждые N эпох)
        if (epoch + 1) % CHECKPOINT_SAVE_INTERVAL == 0 or (epoch + 1) == NUM_EPOCHS:
            checkpoint = {
                'epoch': epoch + 1,
                'gen_state_dict': gen.state_dict(),
                'disc_state_dict': disc.state_dict(),
                'opt_gen_state_dict': opt_gen.state_dict(),
                'opt_disc_state_dict': opt_disc.state_dict(),
                'fixed_noise': fixed_noise,
                'loss_g': loss_gen.item(), 
                'loss_d': loss_disc.item()
            }
            torch.save(checkpoint, f"{OUTPUT_PATH}/checkpoint_epoch_{epoch+1}.pth")
            torch.save(checkpoint, f"{OUTPUT_PATH}/checkpoint_latest.pth") # Перезаписываем последний
            print(f"✅ Checkpoint saved at epoch {epoch+1}")

    print("🏁 Training Finished!")