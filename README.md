# 🎨 DCGAN from Scratch — PyTorch

A clean, from-scratch implementation of **Deep Convolutional Generative Adversarial Networks (DCGAN)** using PyTorch, trained on CIFAR-10.

Based on the [original DCGAN paper](https://arxiv.org/abs/1511.06434) by Radford et al.

## How It Works

A **Generator** learns to create realistic images from random noise, while a **Discriminator** learns to tell real images from fakes. They train adversarially — the generator gets better at fooling the discriminator, and the discriminator gets better at catching fakes.

```
Random Noise → [Generator] → Fake Image
                                  ↓
Real Image → [Discriminator] → Real or Fake?
```

## Architecture

| Component     | Details                                              |
|---------------|------------------------------------------------------|
| Generator     | 5 transposed conv layers, BatchNorm, ReLU, Tanh      |
| Discriminator | 5 conv layers, BatchNorm, LeakyReLU (0.2), Sigmoid   |
| Latent dim    | 100                                                  |
| Image size    | 64×64                                                |
| Optimizer     | Adam (lr=0.0002, β1=0.5, β2=0.999)                  |
| Loss          | Binary Cross-Entropy                                 |
| Epochs        | 25                                                   |

## Requirements

- Python 3.6+
- PyTorch
- torchvision

```bash
pip install torch torchvision
```

## Usage

```bash
cd GANs
mkdir -p results
python dcgan_commented.py
```

CIFAR-10 downloads automatically on first run. Generated samples are saved to `GANs/results/` every 100 training steps.

## Results

**Real samples** (CIFAR-10):

![Real Samples](GANs/results/real_samples.png)

**Generated samples** (epoch 24):

![Generated Samples Epoch 24](GANs/results/fake_samples_epoch_024.png)

The generator progressively learns to produce coherent image structures over 25 epochs of training.

## Known Issues

> These are deprecation warnings from older PyTorch versions. The core logic is correct.

- `transforms.Scale` → use `transforms.Resize` (deprecated since torchvision 0.2)
- `Variable` wrapper → no longer needed (deprecated since PyTorch 0.4)
- `tensor.data[0]` → use `tensor.item()` for scalar access
- No CUDA/GPU support — runs on CPU only

## Project Structure

```
├── GANs/
│   ├── dcgan_commented.py    # Full DCGAN implementation (well-commented)
│   └── results/              # Generated image samples per epoch
├── LICENSE                   # MIT
└── README.md
```

## License

MIT © 2018 [Kaustabh Ganguly](https://github.com/stabgan)
