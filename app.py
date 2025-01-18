from flask import Flask, render_template, request, jsonify
import torch
import numpy as np
from PIL import Image
import io
import base64
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import torch.nn as nn
import logging
import cv2

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_DIMENSION = 1024
TILE_SIZE = 128
OVERLAP = 32
BATCH_SIZE = 4

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.prelu = nn.PReLU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        res = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.prelu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        return x + res

class UpsampleBlock(nn.Module):
    def __init__(self, channels, scale_factor):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels * scale_factor ** 2, kernel_size=3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(scale_factor)
        self.prelu = nn.PReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.pixel_shuffle(x)
        return self.prelu(x)

class Generator(nn.Module):
    def __init__(self, in_channels=3, num_channels=64, num_blocks=16):
        super().__init__()
        self.initial = nn.Sequential(
            nn.Conv2d(in_channels, num_channels, kernel_size=9, padding=4),
            nn.PReLU()
        )
        self.res_blocks = nn.Sequential(*[ResidualBlock(num_channels) for _ in range(num_blocks)])
        self.conv_after_res = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(num_channels)
        self.upsampling = nn.Sequential(
            UpsampleBlock(num_channels, 2),
            nn.Conv2d(num_channels, in_channels, kernel_size=9, padding=4),
            nn.Tanh()
        )

    def forward(self, x):
        initial_out = self.initial(x)
        res_out = self.res_blocks(initial_out)
        res_conv = self.bn(self.conv_after_res(res_out))
        x = torch.add(initial_out, res_conv)
        return self.upsampling(x)

def create_gaussian_weight_matrix(h, w):
    y = np.linspace(-1, 1, h)[:, None]
    x = np.linspace(-1, 1, w)[None, :]
    return np.exp(-4 * (x ** 2 + y ** 2))

def merge_tiles(tiles, positions, weights, input_shape, scale_factor=2):
    b, c, h, w = input_shape
    output_h, output_w = h * scale_factor, w * scale_factor
    output = torch.zeros((b, c, output_h, output_w), device=DEVICE)
    weight_sum = torch.zeros_like(output)

    for idx, (tile, (y, x), weight) in enumerate(zip(tiles, positions, weights)):
        y *= scale_factor
        x *= scale_factor
        _, _, tile_h, tile_w = tile.shape
        weight_resized = torch.nn.functional.interpolate(
            weight[None, None], size=(tile_h, tile_w), mode='bilinear', align_corners=False
        ).squeeze()

        valid_h = min(tile_h, output_h - y)
        valid_w = min(tile_w, output_w - x)
        tile = tile[:, :, :valid_h, :valid_w]
        weight_resized = weight_resized[:valid_h, :valid_w]

        weighted_tile = tile * weight_resized.unsqueeze(0).unsqueeze(0)
        output[:, :, y:y + valid_h, x:x + valid_w] += weighted_tile
        weight_sum[:, :, y:y + valid_h, x:x + valid_w] += weight_resized.unsqueeze(0).unsqueeze(0)

    mask = weight_sum > 0
    output[mask] /= weight_sum[mask]
    return output

def split_image_into_tiles(tensor, tile_size=TILE_SIZE, overlap=OVERLAP):
    b, c, h, w = tensor.shape
    tiles, positions, weights = [], [], []
    effective_tile_size = tile_size - 2 * overlap
    n_tiles_h = max(1, int(np.ceil(h / effective_tile_size)))
    n_tiles_w = max(1, int(np.ceil(w / effective_tile_size)))

    for i in range(n_tiles_h):
        for j in range(n_tiles_w):
            y_start = min(h, i * effective_tile_size)
            x_start = min(w, j * effective_tile_size)
            y_end = min(h, y_start + tile_size)
            x_end = min(w, x_start + tile_size)

            tile = tensor[:, :, y_start:y_end, x_start:x_end]
            weight = create_gaussian_weight_matrix(y_end - y_start, x_end - x_start)
            weight = torch.FloatTensor(weight).to(DEVICE)

            tiles.append(tile)
            positions.append((y_start, x_start))
            weights.append(weight)

    return tiles, positions, weights
import logging
import torch
from typing import List, Tuple

class ImageProcessingLogger:
    def __init__(self, logger_name: str = '__main__', level: int = logging.INFO):
        self.logger = logging.getLogger(logger_name)
        
        # Configure logging if not already configured
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(levelname)s:%(name)s:%(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(level)

    def log_tile_processing(self, tile_index: int, tile: torch.Tensor, position: Tuple[int, int]):
        """
        Log information about a tile being processed.
        
        Args:
            tile_index (int): Index of the current tile
            tile (torch.Tensor): The tile tensor being processed
            position (Tuple[int, int]): The (y, x) position of the tile
        """
        self.logger.info(
            f"Processing tile {tile_index}: shape={tile.shape}, position=({position[0]}, {position[1]})"
        )

    def log_final_output(self, output_tensor: torch.Tensor):
        """
        Log information about the final output tensor.
        
        Args:
            output_tensor (torch.Tensor): The final processed output tensor
        """
        self.logger.info(f"Final output shape: {output_tensor.shape}")
def process_image_in_tiles(tensor, model, tile_size=128, overlap=32, batch_size=4):
    """
    Process an image in tiles with logging.
    
    Args:
        tensor (torch.Tensor): Input image tensor
        model (torch.nn.Module): The model to process tiles
        tile_size (int): Size of tiles
        overlap (int): Overlap between tiles
        batch_size (int): Batch size for processing
    """
    logger = ImageProcessingLogger()
    tiles, positions, weights = split_image_into_tiles(tensor, tile_size, overlap)
    enhanced_tiles = []
    
    # Process tiles in batches
    for batch_idx in range(0, len(tiles), batch_size):
        batch_tiles = tiles[batch_idx:batch_idx + batch_size]
        
        # Log each tile in the batch
        for i, (tile, position) in enumerate(zip(
            batch_tiles, 
            positions[batch_idx:batch_idx + batch_size]
        )):
            tile_index = batch_idx + i
            logger.log_tile_processing(tile_index, tile, position)
            
        # Process the batch
        with torch.no_grad():
            enhanced_batch = [model(tile) for tile in batch_tiles]
            enhanced_tiles.extend(enhanced_batch)
    
    # Merge tiles and log final output
    output = merge_tiles(enhanced_tiles, positions, weights, tensor.shape)
    logger.log_final_output(output)
    
    return output

def preprocess_image(image):
    if image.mode != 'RGB':
        image = image.convert('RGB')

    original_size = image.size
    min_dimension = TILE_SIZE // 2
    if min(image.size) < min_dimension:
        scale = min_dimension / min(image.size)
        new_size = tuple(int(dim * scale) for dim in image.size)
        image = image.resize(new_size, Image.LANCZOS)

    if max(image.size) > MAX_DIMENSION:
        scale = MAX_DIMENSION / max(image.size)
        new_size = tuple(int(dim * scale) for dim in image.size)
        image = image.resize(new_size, Image.LANCZOS)

    img_array = np.array(image).astype(np.float32) / 255.0
    tensor = torch.tensor(img_array).permute(2, 0, 1).unsqueeze(0).to(DEVICE)
    return tensor, img_array, original_size

def postprocess_image(enhanced_tensor, original_size=None):
    enhanced_array = enhanced_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
    enhanced_array = np.clip(enhanced_array, 0, 1)
    enhanced_image = Image.fromarray((enhanced_array * 255).astype(np.uint8))

    if original_size is not None:
        enhanced_image = enhanced_image.resize(original_size, Image.LANCZOS)

    return enhanced_image, enhanced_array

def calculate_metrics(original_array, enhanced_array):
    if original_array.shape != enhanced_array.shape:
        original_array = cv2.resize(
            original_array, 
            (enhanced_array.shape[1], enhanced_array.shape[0]),
            interpolation=cv2.INTER_LINEAR
        )

    psnr_value = psnr(original_array, enhanced_array, data_range=1.0)
    ssim_value = ssim(original_array, enhanced_array, data_range=1.0, multichannel=True, channel_axis=2)
    npsnr_value = 20 * np.log10(1.0 / np.sqrt(np.mean((original_array - enhanced_array) ** 2))) / np.log10(2.0)
    mos_value = (min(psnr_value / 50.0, 1.0) * 0.6 + ssim_value * 0.4) * 5

    return {
        'psnr': float(psnr_value),
        'npsnr': float(npsnr_value),
        'ssim': float(ssim_value),
        'mos': float(max(1.0, min(5.0, mos_value)))
    }

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

model = Generator().to(DEVICE)
checkpoint = torch.load('checkpoint_epoch_100.pth', map_location=DEVICE, weights_only=True)

if 'generator_state_dict' in checkpoint:
    model.load_state_dict(checkpoint['generator_state_dict'])
else:
    model.load_state_dict(checkpoint)
model.eval()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/enhance', methods=['POST'])
def enhance_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    file = request.files['image']
    if file.filename == '' or not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type'}), 400

    input_image = Image.open(io.BytesIO(file.read()))
    processed_input, original_array, original_size = preprocess_image(input_image)
    enhanced_tensor = process_image_in_tiles(processed_input, model)
    enhanced_image, enhanced_array = postprocess_image(enhanced_tensor, original_size)

    metrics = calculate_metrics(original_array, enhanced_array)

    def image_to_base64(image):
        buffered = io.BytesIO()
        image.save(buffered, format="PNG", quality=95)
        return base64.b64encode(buffered.getvalue()).decode()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return jsonify({
        'input_image': f'data:image/png;base64,{image_to_base64(input_image)}',
        'enhanced_image': f'data:image/png;base64,{image_to_base64(enhanced_image)}',
        'metrics': metrics
    })

if __name__ == '__main__':
    app.run(debug=True)
