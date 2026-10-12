import torch
from safetensors.torch import load_file
from torchvision import utils as vutils
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from huggingface_hub import hf_hub_download
import io
import uvicorn

app = FastAPI()

class Generator(torch.nn.Module):
    def __init__(self, nz=100, ngf=64):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            torch.nn.BatchNorm2d(ngf * 8), torch.nn.ReLU(True),
            torch.nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            torch.nn.BatchNorm2d(ngf * 4), torch.nn.ReLU(True),
            torch.nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            torch.nn.BatchNorm2d(ngf * 2), torch.nn.ReLU(True),
            torch.nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            torch.nn.BatchNorm2d(ngf), torch.nn.ReLU(True),
            torch.nn.ConvTranspose2d(ngf, ngf // 2, 4, 2, 1, bias=False),
            torch.nn.BatchNorm2d(ngf // 2), torch.nn.ReLU(True),
            torch.nn.ConvTranspose2d(ngf // 2, 3, 4, 2, 1, bias=False),
            torch.nn.Tanh(),
        )
    def forward(self, z):
        return self.net(z)

# Download model on startup (cached after first run)
model_path = hf_hub_download(
    repo_id="Welmia/welmia-face-1.0-3.6m-base",
    filename="model.safetensors"
)
netG = Generator().eval()
netG.load_state_dict(load_file(model_path))

@app.get("/generate")
def generate_face():
    with torch.no_grad():
        z = torch.randn(1, 100, 1, 1)
        img = netG(z)
    buffer = io.BytesIO()
    vutils.save_image(img, buffer, normalize=True, format='PNG')
    buffer.seek(0)
    return StreamingResponse(buffer, media_type="image/png")

@app.get("/")
def health():
    return {"status": "ok"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
