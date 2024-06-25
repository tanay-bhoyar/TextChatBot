from transformers import AutoModelForSeq2SeqLM,AutoTokenizer
from flask import Flask,render_template
from flask_cors import CORS
from flask import request,jsonify
import json

app=Flask(__name__)
CORS(app)

model_name="facebook/blenderbot-400M-distill"
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
tokenizer=AutoTokenizer.from_pretrained(model_name)
conversation_history=[]

@app.route("/",methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/chatbot',methods=['POST'])
def handle_prompt():
    try:
        data = request.get_json(force=True)
        input_text=data.get("prompt","")
        if not input_text:
            return jsonify({"error":"Prompt is required"}),400
        history_str="\n".join(conversation_history)

        input=tokenizer.encode_plus(history_str,input_text,return_tensors='pt')

        output=model.generate(**input)
        response = tokenizer.decode(output[0],skip_special_tokens=True)
        conversation_history.append(input_text)
        conversation_history.append(response)
        return response

    except Exception as e:
        return jsonify({"error":str(e)}),500

if __name__ == '__main__':
    app.run()