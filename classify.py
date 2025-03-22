import base64
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from io import BytesIO
from PIL import Image
from flask import Flask, request

app = Flask(__name__)

# Load a pretrained ResNet18 model from torchvision (for "hot dog" detection).
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
model.eval()

# Standard ImageNet preprocessing
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# "hot dog" is class index 934 in ImageNet
HOTDOG_INDEX = 934

HTML_FORM = """
<html>
  <body>
    <h1>Hot Dog Checker</h1>
    <form action="/check" method="post" enctype="multipart/form-data">
      <label for="euid">Your EUID:</label><br>
      <input type="text" id="euid" name="euid" required /><br><br>

      <label for="image1">Upload first image:</label>
      <input type="file" name="image1" accept="image/*" required><br><br>

      <label for="image2">Upload second image:</label>
      <input type="file" name="image2" accept="image/*" required><br><br>

      <button type="submit">Check</button>
    </form>
  </body>
</html>
"""

@app.route("/", methods=["GET"])
def home():
    return HTML_FORM

def classify_image(image_bytes):
    """
    Returns a tuple: (verdict_string, base64_string, mime_type)
      - verdict_string: 'It's a hot dog!' or 'Not a hot dog.'
      - base64_string: base64-encoded image for display
      - mime_type: best guess for the image's mime type
    """
    # Convert the raw bytes to base64
    img_base64 = base64.b64encode(image_bytes).decode("utf-8")

    # Attempt to guess the MIME type from the image header
    # If uncertain, default to image/png
    # (Flask's 'mimetype' is typically good enough if the file is valid)
    # We'll handle that logic outside if needed, or just default here
    mime_type = "image/png"

    # Prepare the image for classification
    pil_image = Image.open(BytesIO(image_bytes)).convert('RGB')
    input_tensor = transform(pil_image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(input_tensor)
        predicted_idx = outputs.argmax(dim=1).item()

    # Decide if it's hot dog or not
    if predicted_idx == HOTDOG_INDEX:
        verdict = "Hotdog!"
    else:
        verdict = "Not hotdog."

    return verdict, img_base64, mime_type

@app.route("/check", methods=["POST"])
def check():
    euid = request.form.get("euid", "").strip()
    if not euid:
        return "No EUID provided."

    # Extract the two uploaded images
    if "image1" not in request.files or "image2" not in request.files:
        return "Please upload two images."

    file1 = request.files["image1"]
    file2 = request.files["image2"]

    # Read them as bytes
    file1_data = file1.read()
    file2_data = file2.read()

    # Classify both images
    verdict1, img1_base64, mime1 = classify_image(file1_data)
    verdict2, img2_base64, mime2 = classify_image(file2_data)

    # Build a results page
    # Display EUID, results, and the images
    result_html = f"""
    <html>
      <body>
        <h2>Hello, {euid}!</h2>
        <hr/>
        <table>
            <tr>
                <th>{verdict1}</th>
                <th>{verdict2}</th>
            </tr>
            <tr>
                <td><img src="data:{mime1};base64,{img1_base64}" alt="First Image" height=250/></td>
                <td><img src="data:{mime2};base64,{img2_base64}" alt="Second Image" height=250/></td>
            </tr>
        </table>
        <hr/>
        <a href="/">Try another</a>
      </body>
    </html>
    """
    return result_html

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5050)
