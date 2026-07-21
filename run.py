
from flask import Flask, request
import json
import numpy as np
import plotly.express as px
from analytics.k_means import create_rfm, run_kmeans, dataset, segments
from analytics.timeseries import predict_sales
from analytics.similar_product_prediction import return_similar_products
from analytics.gemma import generate_text_gemma
from flask_cors import CORS
import redis
from waitress import serve
import os


r = redis.Redis(
  host=os.environ.get('REDIS_HOST', 'localhost'),
  port=int(os.environ.get('REDIS_PORT', 6379)),
  password=os.environ.get('REDIS_PASSWORD')
)

app = Flask(__name__)
CORS(app)

def default(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"Object of type {type(o).__name__} is not JSON serializable")

@app.route('/')
def healthcheck():
    return 'Server is running'

@app.route('/create_rfm', methods=['POST'])
def entrypoint():
    try:
        print("begin")
        user_selection = request.json['option']

        # if redis cache is enabled and user_selection is in cache
        if r.get(user_selection):
            return r.get(user_selection)
        print("here")
        df_rfm = create_rfm(user_selection, dataset)

        clusters = run_kmeans(df_rfm)

        clusters['cluster_rank'] = clusters['cluster'].map(segments)

        fig = px.scatter(clusters, x='recency', y='frequency', color='cluster_rank', opacity=0.7, size_max=10, width=500, height=350, title='Clusters')
        response = json.dumps(fig.to_plotly_json(), default=default)
        r.set(user_selection, response, ex=60*60*24*7)
        return response
    except Exception as e:
        print(e)
        return str(e)

@app.route('/predict_sales', methods=['POST'])
def sales_prediction():
    try:
        period = request.json.get('period', 365)
        category = request.json.get('category', None)
        state = request.json.get('state', None)
        if r.get(f'{period}_{category}_{state}'):
            return r.get(f'{period}_{category}_{state}')
        
        response = predict_sales(period, category, state)
        r.set(f'{period}_{category}_{state}', response, ex=60*60*24*7)
        return response
    except Exception as e:
        print(e)
        return str(e)
    
@app.route('/similar_products', methods=['POST'])
def similar_products():
    try:
        # print(request.json)
        product_name = request.json.get('product', None)
        # if r.get(sku):
        #     return r.get(sku)
        # print(product_name)
        response = return_similar_products(product_name)

        r.set(product_name, response, ex=60*60*24*7)
        return response
    except Exception as e:
        print(e)
        return str(e)

    
@app.route('/generate_text', methods=['POST'])
def generate_text():
    try:
        text = request.json.get('text', None)
        # if r.get(text):
        #     return r.get(text)
        response = generate_text_gemma(text)
        r.set(text, response, ex=60*60*24*7)

        return {"text":response}
    except Exception as e:
        print(e)
        return str(e)

if __name__ == '__main__':
    # serve(app, host='0.0.0.0', port=5000)
    app.run(host='0.0.0.0', port=5000, debug=True)