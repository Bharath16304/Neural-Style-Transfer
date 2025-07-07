
import torch
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import matplotlib.pyplot as plt

# Image loading and preprocessing
def load_image(path, max_size=512):
    image = Image.open('content.jpg').convert('RGB')
    size = max(image.size) if max(image.size) < max_size else max_size
    transform = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor()
    ])
    image = transform(image).unsqueeze(0)
    return image.to(device)

# Display function
def imshow(tensor, title=None):
    image = tensor.cpu().clone().squeeze(0)
    image = transforms.ToPILImage()(image)
    plt.imshow(image)
    if title:
        plt.title(title)
    plt.axis('off')
    plt.show()

# Content and style loss calculation
class ContentLoss(torch.nn.Module):
    def __init__(self, target):
        super().__init__()
        self.target = target.detach()

    def forward(self, input):
        self.loss = torch.nn.functional.mse_loss(input, self.target)
        return input

class StyleLoss(torch.nn.Module):
    def __init__(self, target_feature):
        super().__init__()
        self.target = self.gram_matrix(target_feature).detach()

    def gram_matrix(self, input):
        b, c, h, w = input.size()
        features = input.view(b * c, h * w)
        G = torch.mm(features, features.t())
        return G.div(b * c * h * w)

    def forward(self, input):
        G = self.gram_matrix(input)
        self.loss = torch.nn.functional.mse_loss(G, self.target)
        return input

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load images
content_img = load_image("content.jpg")
style_img = load_image("style.jpg")
input_img = content_img.clone()

# Display images
imshow(content_img, title="Content Image")
imshow(style_img, title="Style Image")

# Load VGG19 model
cnn = models.vgg19(pretrained=True).features.to(device).eval()
cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)

# Normalization
class Normalization(torch.nn.Module):
    def __init__(self, mean, std):
        super().__init__()
        self.mean = mean.view(-1, 1, 1)
        self.std = std.view(-1, 1, 1)

    def forward(self, img):
        return (img - self.mean) / self.std

# Build model
content_layers = ['conv_4']
style_layers = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']
model = torch.nn.Sequential()
normalization = Normalization(cnn_normalization_mean, cnn_normalization_std).to(device)

content_losses = []
style_losses = []
model.add_module("normalization", normalization)

i = 0
for layer in cnn.children():
    if isinstance(layer, torch.nn.Conv2d):
        i += 1
        name = f'conv_{i}'
    elif isinstance(layer, torch.nn.ReLU):
        name = f'relu_{i}'
        layer = torch.nn.ReLU(inplace=False)
    elif isinstance(layer, torch.nn.MaxPool2d):
        name = f'pool_{i}'
    else:
        continue

    model.add_module(name, layer)

    if name in content_layers:
        target = model(content_img).detach()
        content_loss = ContentLoss(target)
        model.add_module(f"content_loss_{i}", content_loss)
        content_losses.append(content_loss)

    if name in style_layers:
        target_feature = model(style_img).detach()
        style_loss = StyleLoss(target_feature)
        model.add_module(f"style_loss_{i}", style_loss)
        style_losses.append(style_loss)

# Optimize
optimizer = optim.LBFGS([input_img.requires_grad_()])
style_weight = 1e6
content_weight = 1
num_steps = 300

print("Starting style transfer...")
run = [0]
while run[0] <= num_steps:
    def closure():
        input_img.data.clamp_(0, 1)
        optimizer.zero_grad()
        model(input_img)
        style_score = sum(sl.loss for sl in style_losses)
        content_score = sum(cl.loss for cl in content_losses)
        loss = style_score * style_weight + content_score * content_weight
        loss.backward()

        if run[0] % 50 == 0:
            print(f"Step {run[0]}: Style Loss {style_score.item():.4f} Content Loss {content_score.item():.4f}")
        run[0] += 1
        return loss

    optimizer.step(closure)

input_img.data.clamp_(0, 1)
imshow(input_img, title="Styled Output")
