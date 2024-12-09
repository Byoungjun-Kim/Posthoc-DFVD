from flask import Flask, request, jsonify
from prediction import single_vid

app = Flask(__name__)

@app.route('/fake_prob', methods=['GET'])
def handle_request():
    data = request.get_json()

    youtube_url = data['url']

    net = 'genconvit'
    ed_weight = 'genconvit_ed_inference'
    vae_weight = 'genconvit_vae_inference'
    fp16 = False
    fake_prob = single_vid(ed_weight, vae_weight, youtube_url, net, fp16)
    return jsonify({"fake_prob": fake_prob})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)  # 모든 인터페이스에서 요청을 받도록 설정
