from safetensors.torch import load_file
import torch, os, glob
current_path = "/media/mostafahaggag/D/Projects_ubuntu/APPs/FULL_app/PaddleOCR-Pytorch"
path_of_safe_tensors = os.path.join(current_path, "**/*.safetensors")
for sf_path in glob.glob(path_of_safe_tensors, recursive=True):
    state_dict = load_file(sf_path)
    pth_path = sf_path.replace(".safetensors", ".pth")
    torch.save(state_dict, pth_path)
    print(f"Saved: {pth_path}")