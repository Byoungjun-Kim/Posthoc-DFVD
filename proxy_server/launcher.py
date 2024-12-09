from flask import Flask, request, jsonify
import requests
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# 클라우드 VM에 요청을 전달하고 응답 반환
@app.route('/fake_prob', methods=['POST'])
def handle_request():
    # 외부에서 받은 요청 데이터
    data = request.get_json()

    # 클라우드 VM으로 요청 전달
    vm_url = "http://143.248.48.95:5000/fake_prob"
    try:
        vm_response = requests.get(vm_url, json=data)
        vm_response_data = vm_response.json()
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    # 클라우드 VM에서 받은 응답을 외부로 반환
    return jsonify(vm_response_data)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)

