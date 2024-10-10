
import torch
import torch.nn as nn

class DepthDecoder(nn.Module):
    def __init__(self, input_channels, output_channels1, output_channels2):
        super().__init__()
        
        self.decoder = nn.Sequential(
            nn.Conv2d(input_channels, input_channels // 2, kernel_size=3, stride=1, padding=1),
            nn.ConvTranspose2d(
                in_channels=input_channels // 2,
                out_channels=input_channels // 2,
                kernel_size=2,
                stride=2,
                padding=0,
                bias=True,
            ),
            nn.Conv2d(input_channels // 2, output_channels1, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(output_channels1, output_channels2, kernel_size=1, stride=1, padding=0),
            nn.ReLU()
        )
    
    def forward(self, x):
        return self.decoder(x)


if __name__=="__main__":
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Input configuration
    input_channels = 64
    output_channels1 = 32
    output_channels2 = 16
    
    # Create an instance of the CustomConvDecoder and move it to the selected device
    model = DepthDecoder(input_channels, output_channels1, output_channels2).to(device)
    
    # Create a dummy input tensor of size [batch_size, input_channels, height, width]
    # Example: batch size = 8, height = 64, width = 64, and move it to the device
    input_tensor = torch.randn(8, input_channels, 64, 64).to(device)
    
    # Pass the input through the model
    output = model(input_tensor)
    
    # Print the output size for verification
    print("Output shape:", output.shape)
    
    # Check the output shape to make sure it matches the expected size
    # Expecting: [batch_size, output_channels2, height_out, width_out]
    # The height and width should have doubled due to the ConvTranspose2d layer
    expected_output_shape = (8, output_channels2, 128, 128)
    assert output.shape == expected_output_shape, f"Expected {expected_output_shape}, but got {output.shape}"

    print("Test passed successfully on device:", device)