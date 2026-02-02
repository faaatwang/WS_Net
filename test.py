import torch
from PIL import Image
import torchvision.transforms as T
from matplotlib import pyplot as plt
from torchvision import transforms
from model.WS_Net import WS_Net
import argparse

def show_image(x, title='Image'):

    x = x.to('cpu')

    mean = torch.tensor([0.485, 0.456, 0.406])
    std = torch.tensor([0.229, 0.224, 0.225])
    x = x * std.view(1, 3, 1, 1) + mean.view(1, 3, 1, 1)

    x = x.permute(0, 2, 3, 1)

    x = (x * 255).type(torch.uint8)

    plt.figure()
    plt.imshow(x[0].numpy())
    plt.title(title)
    plt.axis('off')
    plt.show()

def get_args():
    parser = argparse.ArgumentParser(description='WS_Net inference')

    parser.add_argument(
        '--img_path',
        type=str,
        required=True,
        help='Path to input image'
    )

    parser.add_argument(
        '--model_path',
        type=str,
        required=True,
        help='Path to trained model weights'
    )

    return parser.parse_args()

def main(img_path, model_path):
    # load image to tensor
    normalizer = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                      std=[0.229, 0.224, 0.225])
    transform = T.Compose([
        T.Resize((256, 256)),
        T.ToTensor(),
        normalizer
    ])
    img = Image.open(img_path)
    img = img.convert("RGB")
    x = transform(img).unsqueeze(0).cuda()

    show_image(x, title='Original Image')

    # load WS_Net
    model = WS_Net().cuda()
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # purify
    with torch.no_grad():
        x = model(None, x, 0)

    show_image(x, title='Purified Image')

def main_wrapper():
    args = get_args()

    main(
        img_path=args.img_path,
        model_path=args.model_path
    )

if __name__ == '__main__':
    main_wrapper()

