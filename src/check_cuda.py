import torch
import subprocess
import sys
from typing import Optional, Tuple

def get_nvidia_smi_version() -> Optional[str]:
    """Get CUDA version from nvidia-smi if available"""
    try:
        output = subprocess.check_output(["nvidia-smi"], universal_newlines=True)
        for line in output.split('\n'):
            if 'CUDA Version' in line:
                return line.split('CUDA Version: ')[1].strip()
    except (subprocess.SubprocessError, FileNotFoundError):
        return None
    return None

def get_cuda_info() -> Tuple[bool, Optional[str], Optional[str]]:
    """Get CUDA availability and version information"""
    cuda_available = torch.cuda.is_available()
    torch_cuda_version = None
    nvidia_cuda_version = None
    
    if cuda_available:
        # Get PyTorch's CUDA version
        torch_cuda_version = torch.version.cuda
        # Get system CUDA version from nvidia-smi
        nvidia_cuda_version = get_nvidia_smi_version()
        
    return cuda_available, torch_cuda_version, nvidia_cuda_version

def main():
    cuda_available, torch_cuda, nvidia_cuda = get_cuda_info()
    
    print("\n=== CUDA Environment Check ===")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {cuda_available}")
    
    if cuda_available:
        print(f"PyTorch CUDA version: {torch_cuda}")
        print(f"System CUDA version: {nvidia_cuda}")
        print(f"Current CUDA device: {torch.cuda.current_device()}")
        print(f"CUDA device name: {torch.cuda.get_device_name(0)}")
        print(f"CUDA device count: {torch.cuda.device_count()}")
        print("\nCUDA device properties:")
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"\nDevice {i}: {props.name}")
            print(f"  Compute capability: {props.major}.{props.minor}")
            print(f"  Total memory: {props.total_memory / 1024**3:.2f} GB")
            print(f"  Multi-processor count: {props.multi_processor_count}")
    else:
        print("\nWarning: CUDA is not available. The system will fall back to CPU.")
        print("For optimal performance, please ensure CUDA is properly installed.")

if __name__ == "__main__":
    main()