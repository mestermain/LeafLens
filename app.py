import os
import uuid
import numpy as np
import io
import base64
from PIL import Image
import cv2
from flask import Flask, render_template, request, jsonify

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.path.join('static', 'uploads')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASS_NAMES_4 = ['healthy', 'nitrogen-N', 'phosphorus-P', 'potasium-K']
CLASS_INFO = {
    'healthy': {'title': 'Healthy Leaf', 'desc': 'Vibrant green color, optimal photosynthetic activity.', 'action': 'Maintain standard irrigation and fertilization schedule.'},
    'nitrogen-N': {'title': 'Nitrogen Deficiency (N)', 'desc': 'General yellowing (chlorosis) of older leaves, stunted growth.', 'action': 'Apply nitrogen-rich fertilizer (e.g., Urea, Ammonium Nitrate).'},
    'phosphorus-P': {'title': 'Phosphorus Deficiency (P)', 'desc': 'Dark green leaves with purplish margins, poor root development.', 'action': 'Apply phosphate fertilizers (e.g., Superphosphate).'},
    'potasium-K': {'title': 'Potassium Deficiency (K)', 'desc': 'Yellowing and necrosis (browning and dying) of leaf margins and tips.', 'action': 'Apply potassium sulfate or muriate of potash.'},
}

# Image Transformations
std_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

class StandardGradCAMViT:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output
        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0]
        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate(self, input_tensor, target_class=None):
        self.model.eval()
        self.model.zero_grad()
        output = self.model(input_tensor)
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        score = output[0, target_class]
        score.backward(retain_graph=True)
        
        grads = self.gradients.data.cpu().numpy()[0]
        acts = self.activations.data.cpu().numpy()[0]
        
        if grads.ndim == 2:
            grads_patch = grads[1:, :]
            acts_patch = acts[1:, :]
            weights = np.mean(grads_patch, axis=0)
            cam = np.dot(acts_patch, weights)
            num_patches = cam.shape[0]
            grid_size = int(np.sqrt(num_patches))
            cam = cam.reshape(grid_size, grid_size)
        else:
            weights = np.mean(grads, axis=(1, 2))
            cam = np.sum(weights[:, np.newaxis, np.newaxis] * acts, axis=0)
            
        cam = np.maximum(cam, 0)
        if cam.max() > 0:
            cam = cam / cam.max()
        return cam, output.detach().cpu().numpy()[0], target_class

def load_4_class_vit():
    try:
        print(f"Loading 4-Class ViT model onto {device}...")
        vit = models.vit_b_16(weights=None)
        vit.heads.head = nn.Linear(vit.heads.head.in_features, 4)
        path = os.path.join(os.path.dirname(__file__), "models", "vit_coleaf.pth")
        if os.path.exists(path):
            vit.load_state_dict(torch.load(path, map_location=device))
            print("Model weights loaded successfully.")
        else:
            print(f"Warning: Model weights not found at {path}")
        vit.to(device)
        vit.eval()
        target_layer = vit.encoder.layers[-1].ln_1
        cam_engine = StandardGradCAMViT(vit, target_layer)
        return vit, cam_engine
    except Exception as e:
        print(f"Failed to load ViT model: {e}")
        return None, None

vit_model, cam_engine = load_4_class_vit()

def validate_leaf_image(pil_img):
    try:
        cv_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        hsv = cv2.cvtColor(cv_img, cv2.COLOR_BGR2HSV)
        
        lower_green = np.array([15, 15, 15])
        upper_green = np.array([95, 255, 255])
        lower_brown = np.array([4, 15, 15])
        upper_brown = np.array([25, 255, 220])

        mask_green = cv2.inRange(hsv, lower_green, upper_green)
        mask_brown = cv2.inRange(hsv, lower_brown, upper_brown)
        combined_mask = cv2.bitwise_or(mask_green, mask_brown)

        total_pixels = cv_img.shape[0] * cv_img.shape[1]
        plant_pixel_ratio = np.count_nonzero(combined_mask) / max(total_pixels, 1)
        
        gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()

        if plant_pixel_ratio < 0.05:
            return False, "The uploaded image does not appear to contain a coffee leaf. Color spectrum lacks plant tissue tones."
        if laplacian_var < 5.0:
            return False, "The image is too blank or blurry to detect leaf vein structures."
        return True, "Valid leaf image"
    except Exception:
        return True, "Valid leaf image"

def to_base64(img_array):
    _, buffer = cv2.imencode('.png', img_array)
    b64_str = base64.b64encode(buffer).decode('utf-8')
    return f"data:image/png;base64,{b64_str}"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'success': False, 'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'success': False, 'error': 'No selected file'}), 400

    file_ext = os.path.splitext(file.filename)[1].lower()
    if file_ext not in ['.jpg', '.jpeg', '.png']:
        return jsonify({'success': False, 'error': 'Invalid file format. Upload JPG or PNG.'}), 400

    try:
        file_bytes = file.read()
        pil_img = Image.open(io.BytesIO(file_bytes)).convert('RGB')
        
        # 1. LEAF VALIDATION
        is_valid_leaf, validation_reason = validate_leaf_image(pil_img)
        if not is_valid_leaf:
            return jsonify({
                'success': False, 
                'error': f'Constraint Failed: {validation_reason} Only images containing leaves are acceptable.'
            }), 400

        if vit_model is None or cam_engine is None:
            return jsonify({'success': False, 'error': 'ViT Model could not be loaded locally.'}), 500

        # 2. LOCAL AI INFERENCE
        input_std = std_transform(pil_img).unsqueeze(0).to(device)
        cam, logits, pred_idx = cam_engine.generate(input_std)
        probs = torch.softmax(torch.tensor(logits), dim=0).numpy()
        
        predicted_class = CLASS_NAMES_4[pred_idx]
        confidence = float(probs[pred_idx]) * 100.0

        # 3. GRAD-CAM OVERLAY
        cv_img_rgb = np.array(pil_img)
        h, w, _ = cv_img_rgb.shape
        cam_resized = cv2.resize(cam, (w, h))
        heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
        heatmap_rgb = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        overlay = cv2.addWeighted(cv_img_rgb, 0.55, heatmap_rgb, 0.45, 0)
        
        # 4. JSON RESPONSE
        info = CLASS_INFO.get(predicted_class, {})
        
        prob_list = []
        for idx, cls in enumerate(CLASS_NAMES_4):
            prob_list.append({
                'class': cls,
                'title': CLASS_INFO.get(cls, {}).get('title', cls),
                'probability': round(float(probs[idx]) * 100, 2)
            })
        prob_list.sort(key=lambda x: x['probability'], reverse=True)

        return jsonify({
            'success': True,
            'model_mode': '4_class',
            'predicted_class': predicted_class,
            'class_title': info.get('title', predicted_class),
            'confidence': round(confidence, 2),
            'description': info.get('desc', 'Coffee leaf deficiency classification.'),
            'action': info.get('action', 'Consult agronomy specialist.'),
            'original_image_base64': to_base64(cv2.cvtColor(cv_img_rgb, cv2.COLOR_RGB2BGR)),
            'gradcam_image_base64': to_base64(cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)),
            'probabilities': prob_list,
            'models_summary': {'Vision Transformer (ViT)': {'class': predicted_class, 'confidence': round(confidence, 2)}}
        }), 200

    except Exception as err:
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(err)}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    print(f"\nStarting Local LeafLens Flask API on port {port}...")
    app.run(host='0.0.0.0', port=port, debug=False)
