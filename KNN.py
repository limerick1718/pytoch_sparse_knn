import torch
import numpy as np
import sparse
from sklearn.feature_extraction.text import CountVectorizer

device = torch.device("cuda:2")
def knn(x, queries, k=10):
    pdist = torch.nn.PairwiseDistance(p=2)
    # x = X_1_tag_tf
    # queries = X_2_tag_tf
    results = []
    for query in queries:
        # iter = 0
        # query = queries[iter].todense()
        top_dist = {}
        base_cnt = 0
        batch_size = 10000
        while True:
            if base_cnt + batch_size > x.shape[0]:
                batch_x = x[base_cnt:x.shape[0]]
            else:
                batch_x = x[base_cnt:batch_size + base_cnt]
            batch_x = batch_x.todense()
            zeros = torch.tensor(np.zeros_like(batch_x)).to(device)
            batch_x = torch.tensor(batch_x).to(device)
            query = torch.tensor(query).to(device)
            diff = batch_x - query
            dists = pdist(diff, zeros)
            dist, col = torch.topk(dists, k=k, dim=0, largest=False)
            dist = dist.cpu().tolist()
            col = col.cpu().tolist()
            for i in range(5):
                top_dist[col[i] + + base_cnt] = dist[i]
            base_cnt += batch_size
            if base_cnt > x.shape[0]:
                break
        sorted_dist = sorted(top_dist.items(), key=lambda d:d[1])[:5]
        labels = []
        for pair in sorted_dist:
            col = pair[0]
            # label = y.iloc[col]
            # labels.append(label)
        # results.append(np.mean(labels))
    return results

if __name__ == "__main__":
    vectorizer = CountVectorizer(stop_words='english', dtype=np.float32).fit(X_1)
    X_1_tf = sparse.COO.from_scipy_sparse(vectorizer.transform(X_1))
    X_2_tf = sparse.COO.from_scipy_sparse(vectorizer.transform(X_2))
    knn(X_1_tf, X_2_tf, k=10)
