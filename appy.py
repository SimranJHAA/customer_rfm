from flask import Flask, request, jsonify, send_from_directory
import os
from flask_cors import CORS  
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import silhouette_score 
import os

app = Flask(__name__)
CORS(app)  

@app.route('/segment', methods=['POST'])
def segment_customers():
    try:
        file = request.files['file']
        df = pd.read_csv(file, encoding="unicode_escape")
        df.dropna(inplace=True)
        df['CustomerID'] = df['CustomerID'].astype(str)
        df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'], format='mixed')

        # RFM
        df['Amount'] = df['Quantity'] * df['UnitPrice']
        rfm_m = df.groupby('CustomerID')['Amount'].sum().reset_index()
        rfm_m.columns = ['CustomerID', 'Amount']

        rfm_f = df.groupby('CustomerID')['InvoiceNo'].count().reset_index()
        rfm_f.columns = ['CustomerID', 'Frequency']

        max_date = df['InvoiceDate'].max()
        df['Diff'] = (max_date - df['InvoiceDate']).dt.days
        rfm_p = df.groupby('CustomerID')['Diff'].min().reset_index()
        rfm_p.columns = ['CustomerID', 'Recency']

        rfm_df = rfm_m.merge(rfm_f, on='CustomerID').merge(rfm_p, on='CustomerID')

        # Outlier removal
        Q1 = rfm_df['Amount'].quantile(0.05)
        Q3 = rfm_df['Amount'].quantile(0.95)
        IQR = Q3 - Q1
        rfm_df = rfm_df[
            (rfm_df['Amount'] >= Q1 - 1.5 * IQR) &
            (rfm_df['Amount'] <= Q3 + 1.5 * IQR)
        ]

        # Scaling
        scaler = MinMaxScaler()
        rfm_scaled = scaler.fit_transform(rfm_df[['Amount', 'Frequency', 'Recency']])
        rfm_scaled_df = pd.DataFrame(rfm_scaled, columns=['Amount', 'Frequency', 'Recency'])
        # Clustering (n_clusters=3 as in your notebook)
        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10, max_iter=50) 
        cluster_labels = kmeans.fit_predict(rfm_scaled_df)
        rfm_df['Cluster'] = cluster_labels
        #SILHOUETTE SCORE
        silhouette_avg = silhouette_score(rfm_scaled_df, cluster_labels)
        print(f"Silhouette Score for 3 clusters: {silhouette_avg:.4f}")
		
        # Cluster percentages
        cluster_counts = rfm_df['Cluster'].value_counts().sort_index()
        cluster_percent = (cluster_counts / len(rfm_df)) * 100

        # Convert to list of dicts for JSON
        result_data = rfm_df.to_dict(orient='records')
        cluster_stats = cluster_percent.to_dict()
        return jsonify({
            "success": True,
            "data": result_data,
            "cluster_percentages": cluster_stats,
            "silhouette_score": round(silhouette_avg, 4),
            "total_customers": len(rfm_df)
        })

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400
		
# Serve index.html at root URL
@app.route('/')
def serve_index():
    return send_from_directory('.', 'index.html')

# Serve static files (if any)
@app.route('/<path:path>')
def serve_static(path):
    if os.path.exists(path):
        return send_from_directory('.', path)
    else:
        return send_from_directory('.', 'index.html')
		
if __name__ == '__main__':

    app.run(debug=True, host='0.0.0.0', port=5001)



