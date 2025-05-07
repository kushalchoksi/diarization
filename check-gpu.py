import torch
import sys

def check_gpu():
    """
    Checks for CUDA availability and prints GPU details if found.
    """
    print(f"--- PyTorch GPU Check ---")
    print(f"Python Version: {sys.version}")
    print(f"PyTorch Version: {torch.__version__}")

    # Check if CUDA (for NVIDIA GPUs) is available
    is_available = torch.cuda.is_available()

    print(f"\nCUDA Available: {is_available}")

    if is_available:
        # Get the number of GPUs available
        gpu_count = torch.cuda.device_count()
        print(f"Number of GPUs Available: {gpu_count}")

        # Get the name of the primary GPU (device 0)
        try:
            gpu_name = torch.cuda.get_device_name(0)
            print(f"GPU Name (Device 0): {gpu_name}")

            # Optional: Print details for all GPUs if more than one
            if gpu_count > 1:
                 print("\nDetails for all GPUs:")
                 for i in range(gpu_count):
                     print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")

            # Optional: Perform a small tensor operation on the GPU
            try:
                print("\nAttempting a small operation on GPU...")
                # Create a tensor on the CPU
                tensor_cpu = torch.tensor([1.0, 2.0])
                print(f"  Tensor on CPU: {tensor_cpu}")
                # Move the tensor to the GPU
                tensor_gpu = tensor_cpu.to('cuda')
                print(f"  Tensor moved to GPU: {tensor_gpu}")
                # Perform a simple operation
                result_gpu = tensor_gpu * 2
                print(f"  Result of operation on GPU: {result_gpu}")
                # Move result back to CPU for printing (optional)
                result_cpu = result_gpu.to('cpu')
                print(f"  Result moved back to CPU: {result_cpu}")
                print("  GPU operation successful.")
            except Exception as e:
                print(f"  Error during GPU operation test: {e}")

        except Exception as e:
            print(f"Error getting GPU details: {e}")

    else:
        print("\nCUDA is not available. PyTorch will use the CPU.")
        print("Ensure you have the correct PyTorch version installed for your CUDA toolkit.")
        print("Check NVIDIA drivers and CUDA installation.")

    print(f"\n--- Check Complete ---")

if __name__ == "__main__":
    check_gpu()
