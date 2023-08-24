from flask import Flask, jsonify, request
import io, json, base64
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

app = Flask(__name__)

def transform_image(image_bytes):
    my_transforms = transforms.Compose([transforms.Resize(255), 
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
                                        )])
    image = Image.open(io.BytesIO(image_bytes))
    return my_transforms(image).unsqueeze(0)
model = models.densenet121(weights='IMAGENET1K_V1')
model.eval()

imgnet_class_index = json.load(open('./imagenet_class_index.json'))

def get_prediction(image_bytes):
    tensor = transform_image(image_bytes=image_bytes)
    outputs = model.forward(tensor)
    _, y_hat = outputs.max(1)
    predicted_idx = str(y_hat.item())
    # return predicted_idx
    return imgnet_class_index[predicted_idx]

def get_multi_predict(image_bytes):
    tensor = transform_image(image_bytes=image_bytes)
    outputs = model.forward(tensor)
    _, y_hat = outputs.topk(3)
    predicts = y_hat.tolist()[0]
    multi_predict = dict([[n, imgnet_class_index[str(i)]] for n, i in enumerate(predicts, start=1)])
    # return predicted_idx
    return multi_predict



@app.route('/')
def hello():
    return 'Hello World!'

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        file = json.load(io.BytesIO(request.data))
        # class_id, class_name = get_prediction(image_bytes=img_bytes)
        # return jsonify({'class_id': class_id, 'class_name': class_name})
        multi_predict = get_multi_predict(image_bytes=base64.b64decode(file['file']))
        # print(request.json())
        return jsonify(multi_predict)
    
if __name__ == '__main__':
    app.run(port=8080, debug=True)