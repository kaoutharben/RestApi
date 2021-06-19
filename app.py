from flask import Flask,request,jsonify
import recommendation_model
import sys

app = Flask(__name__)
        
@app.route('/', methods=['GET','POST'])
def index():
        if request.method == 'POST':
                data=request.get_json(force=True)
                try:
                        app = data["titles"]
                        res = recommendation_model.recommendations(app)
                        return res
                except:
                        return ("error",sys.exc_info()[0])
        else:
                return 'hello world'

                   
   
   

#if __name__=='__main__':
#        app.run(debug=True)