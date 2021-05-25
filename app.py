from flask import Flask,request,jsonify
from flask_cors import CORS
import recommendation_model

app = Flask(__name__)
CORS(app) 
        
@app.route('/', methods=['GET'])
def recommend_movies():
        app=request.get_json()['titles']
        res = recommendation_model.recommendations(app)
        return res
        
        

if __name__=='__main__':
        app.run(debug=True)