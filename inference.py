import os
import argparse
import torch
import cv2
import numpy as np
from model import EfficientSegNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()

parser.add_argument("--in_dir", required=True)
parser.add_argument("--out_dir", required=True)

args = parser.parse_args()

os.makedirs(args.out_dir, exist_ok=True)

model = EfficientSegNet().to(device)
model.load_state_dict(torch.load("best_model.pth",map_location=device))
model.eval()

for img_name in os.listdir(args.in_dir):

    path = os.path.join(args.in_dir,img_name)

    img = cv2.imread(path)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    img = cv2.resize(img,(300,300))

    img = img/255.0

    img = np.transpose(img,(2,0,1))

    tensor = torch.tensor(img).float().unsqueeze(0).to(device)

    with torch.no_grad():
        pred = model(tensor)

    mask = torch.argmax(pred,dim=1).squeeze().cpu().numpy()

    binary = (mask>0).astype(np.uint8)*255

    out_name = img_name

    cv2.imwrite(
        os.path.join(args.out_dir,out_name),
        binary
    )