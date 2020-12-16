from flask import Flask,render_template,request
import pickle
import matplotlib.pyplot as plt
import os
from io import BytesIO
import base64


app = Flask(__name__)

file=open('model.pkl','rb')
clf=pickle.load(file)
file.close()


@app.route('/',methods=["GET","POST"])
def hello_world():
    if request.method=="POST":
        myDict=(request.form)
        weekend=int(myDict['Weekend'])
        Weather=int(myDict['Weather'])
        time=int(myDict['time'])
        requirement=[weekend,Weather,time]
        trafficProb=clf.predict_proba([requirement])[0][1]
        print(trafficProb)

        
        xyz=trafficProb*100
        
        labels = 'Congestion', 'Non congestion'
        sizes = [xyz, 100-xyz]
        explode = (0, 0.1)
        fig1, ax1 = plt.subplots()
        ax1.pie(sizes, explode=explode,labels=labels, autopct='%1.1f%%',shadow=True, startangle=90)
        ax1.axis('equal') 
        plt.savefig('C://Users//KIIT//Desktop//Traffic Prediction//static//abc.png')
        
        figfile = BytesIO()
        plt.savefig(figfile, format='png')
        figfile.seek(0)
        figdata_png = base64.b64encode(figfile.getvalue()).decode('ascii')
        result = figdata_png
        
        

        if xyz>90:
            a="very intense traffic"
        elif xyz>70 and xyz<89:
            a="high traffic"
        elif xyz>40 and xyz<69:
            a="Moderate traffic"
        elif xyz>10 and xyz<39:
            a="light traffic"
        elif xyz<10:
            a="no traffic sir"

        
        

           
            
        return render_template('show.html',traf=round(trafficProb*100),a=a,result=figdata_png)
        
    return render_template('index.html')



if __name__ == "__main__":
    app.run(debug=True)