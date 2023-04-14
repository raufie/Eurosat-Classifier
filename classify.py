from ResNet import ResNet
import torch
import torchvision.transforms as transforms
from PIL import Image
import json
import sys

image_location = "./test.jpg"

try:
    args = sys.argv
    for i, arg in enumerate(args):
        if arg == "-file":
            image = args[i+1]
            f = open(image,"r")
            image_location = image
            break
except Exception as E:
    print("Error finding the file")
    print(str(E))
    print("\033[1;32;40mLoading default test.jpg\033[1;37;40m")



labels = {}
with open("./labels.json",'r') as f:
    labels = json.load(f)

index_to_label = {labels[k]:k for k in labels}

Device = torch.device("cpu")
if torch.cuda.is_available():
    print("CUDA device avalable")
    Device = torch.device("cuda")

transformToTensor = transforms.Compose([
transforms.Resize((64,)),
transforms.CenterCrop((64, 64)),
transforms.ToTensor()
]
)

model = ResNet(10)
model = torch.nn.DataParallel(model)
model.to(Device)
model.load_state_dict(torch.load("./eurosat_1.pt",map_location=Device) )
model.eval()
# open image
img = Image.open(image_location)

imgTensor = transformToTensor(img)
imgTensor = imgTensor.unsqueeze(0)
# print(imgTensor.shape)
out = model(imgTensor)
# print(out.shape)
probs = torch.softmax(out, dim=1)
# print(probs.argmax())
print(index_to_label[probs.argmax().item()], "{:.4f}".format(probs.max().item()*100)+" %")

