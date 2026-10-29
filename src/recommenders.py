"""MovieLens ranking models, using original user_id/movie_id throughout public APIs."""
from pathlib import Path
import hashlib
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.preprocessing import normalize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split


def fingerprint(frame):
    cols = ['user_id', 'movie_id', 'rating']
    ordered = frame[cols].astype('float64').sort_values(['user_id', 'movie_id']).reset_index(drop=True)
    return hashlib.sha256(pd.util.hash_pandas_object(ordered, index=False).values.tobytes()).hexdigest()


def validate(frame):
    required = ['user_id', 'movie_id', 'rating']
    if not set(required).issubset(frame):
        raise ValueError('Cần user_id, movie_id, rating theo notebook 01 mới.')
    if frame.empty or frame[required].isna().any().any():
        raise ValueError('Dữ liệu rỗng hoặc có giá trị thiếu.')
    if not np.isfinite(frame[required].to_numpy(dtype=float)).all() or not frame.rating.between(1, 5).all():
        raise ValueError('ID/rating không hợp lệ; rating phải trong [1, 5].')
    if frame.duplicated(['user_id', 'movie_id']).any():
        raise ValueError('Trùng cặp user-movie; cần xử lý trước khi chia tập.')


def load_data(root):
    root = Path(root)
    train = pd.read_csv(root / 'data/processed/train.csv')
    test = pd.read_csv(root / 'data/processed/test.csv')
    movies = pd.read_csv(root / 'data/raw/movies.dat', sep='::', engine='python',
                         names=['movie_id', 'title', 'genres'], encoding='latin-1')
    validate(train)
    validate(test)
    if set(zip(train.user_id, train.movie_id)) & set(zip(test.user_id, test.movie_id)):
        raise ValueError('Train và test trùng cặp user-movie.')
    if movies.movie_id.duplicated().any():
        raise ValueError('Metadata trùng movie_id.')
    if not set(train.movie_id).issubset(movies.movie_id):
        raise ValueError('Thiếu metadata cho phim train.')
    return train, test, movies


def split_validation(train):
    """Inner 90/10 split; never tune on the outer test from notebook 01."""
    return train_test_split(train, test_size=.1, random_state=42)


class Popularity:
    def fit(self, train, movies=None):
        validate(train)
        self.train = train[['user_id', 'movie_id', 'rating']].copy()
        self.users = np.sort(train.user_id.unique())
        self.items = np.sort(train.movie_id.unique())
        self.user_pos = {x: i for i, x in enumerate(self.users)}
        self.item_pos = {x: i for i, x in enumerate(self.items)}
        self.seen = train.groupby('user_id').movie_id.apply(set).to_dict()
        self.R = csr_matrix((train.rating.to_numpy(dtype=np.float32),
                             (train.user_id.map(self.user_pos), train.movie_id.map(self.item_pos))),
                            shape=(len(self.users), len(self.items)))
        self.B = self.R.copy()
        self.B.data = (self.B.data >= 4).astype(np.float32)
        self.B.eliminate_zeros()
        self.pop = np.asarray(self.B.sum(axis=0)).ravel()
        return self

    def scores(self, user):
        return self.pop.copy()

    def ranked(self, scores, n=10, exclude=()):
        scores = np.asarray(scores).ravel()
        if len(scores) != len(self.items) or not np.isfinite(scores).all():
            raise ValueError('Vector điểm không hợp lệ.')
        # Deterministic tie-breaking, unlike unspecified argsort ties in original notebooks.
        order = np.lexsort((self.items, -self.pop, -scores))
        excluded = set(exclude)
        return [(int(self.items[i]), float(scores[i])) for i in order
                if self.items[i] not in excluded][:max(0, int(n))]

    def recommend(self, user, n=10):
        return [i for i, _ in self.ranked(self.scores(user), n, self.seen.get(user, set()))]


def knn_graph(matrix, k, block_size=128):
    """Exact cosine kNN in blocks: no full N×N similarity matrix in memory.
    Keep positive neighbors only; remove self; ties use row index ascending.
    """
    vectors = normalize(matrix, axis=1).tocsr()
    size = vectors.shape[0]
    k = min(k, max(0, size-1))
    row_ids, col_ids, weights = [], [], []
    for start in range(0, size, block_size):
        stop = min(start + block_size, size)
        similarities = (vectors[start:stop] @ vectors.T).toarray()
        similarities[np.arange(stop-start), np.arange(start, stop)] = 0
        top = np.argsort(-similarities, axis=1, kind='stable')[:, :k]
        values = np.take_along_axis(similarities, top, axis=1)
        positive = values > 0
        rows = np.broadcast_to(np.arange(start, stop)[:, None], top.shape)
        row_ids.extend(rows[positive].tolist())
        col_ids.extend(top[positive].tolist())
        weights.extend(values[positive].tolist())
    return csr_matrix((weights, (row_ids, col_ids)), shape=(size, size), dtype=np.float32)


class CollaborativeFiltering(Popularity):
    def __init__(self, mode='item', neighbors=20):
        if mode not in ['user', 'item'] or neighbors < 1:
            raise ValueError('mode=user/item và neighbors >=1')
        self.mode, self.neighbors = mode, int(neighbors)

    def fit(self, train, movies=None):
        super().fit(train)
        self.graph = knn_graph(self.B if self.mode == 'user' else self.B.T.tocsr(), self.neighbors)
        return self

    def scores(self, user):
        if user not in self.user_pos:
            return self.pop.copy()
        u = self.user_pos[user]
        if self.mode == 'user':
            scores = (self.graph[u] @ self.B).toarray().ravel()
        else:
            # Each liked seed movie contributes to its k most similar neighbors.
            scores = (self.B[u] @ self.graph).toarray().ravel()
        return scores if np.any(scores > 0) else self.pop.copy()


def genre_tokens(text):
    return [g.strip().lower() for g in text.split('|') if g.strip()]


class ContentBased(Popularity):
    def fit(self, train, movies):
        super().fit(train)
        self.catalog = movies.set_index('movie_id').reindex(self.items)
        if self.catalog.title.isna().any():
            raise ValueError('Thiếu metadata phim.')
        genres = self.catalog.genres.fillna('').astype(str)
        self.vectorizer = TfidfVectorizer(tokenizer=genre_tokens, token_pattern=None, lowercase=False)
        self.X = self.vectorizer.fit_transform(genres).tocsr()
        # Separate title/genre text index for search only, never for user ranking.
        title = self.catalog.title.str.replace(r'\(\d{4}\)\s*$', '', regex=True)
        text = title + ' ' + genres.str.replace('|', ' ', regex=False)
        self.search_vectorizer = TfidfVectorizer(ngram_range=(1, 2), token_pattern=r"(?u)\b[\w'-]+\b")
        self.search_X = self.search_vectorizer.fit_transform(text).tocsr()
        return self

    def scores(self, user):
        if user not in self.user_pos:
            return self.pop.copy()
        row = self.R[self.user_pos[user]].copy()
        row.data = np.maximum(row.data - 3, 0)
        row.eliminate_zeros()
        if row.nnz == 0:
            return self.pop.copy()
        profile = normalize(row @ self.X)
        scores = (self.X @ profile.T).toarray().ravel()
        return scores if np.any(scores > 0) else self.pop.copy()

    def similar(self, movie, n=10):
        if movie not in self.item_pos:
            return []
        scores = (self.X @ self.X[self.item_pos[movie]].T).toarray().ravel()
        return [(i, s) for i, s in self.ranked(scores, n, [movie]) if s > 0]

    def search(self, query, n=10):
        vector = self.search_vectorizer.transform([str(query)])
        if vector.nnz == 0:
            return []
        scores = (self.search_X @ vector.T).toarray().ravel()
        return [(i, s) for i, s in self.ranked(scores, n) if s > 0]


def evaluate(model, heldout, k=10):
    """Warm-start filtering matches notebook 04; threshold >=4, macro per-user.
    Precision denominator is k even if fewer recommendations. Reject overlap.
    """
    if k < 1:
        raise ValueError('k >= 1')
    if set(zip(model.train.user_id, model.train.movie_id)) & set(zip(heldout.user_id, heldout.movie_id)):
        raise ValueError('Leakage: train và heldout trùng cặp.')
    known = heldout[heldout.user_id.isin(model.users) & heldout.movie_id.isin(model.items)]
    truth = known[known.rating >= 4].groupby('user_id').movie_id.apply(set)
    rows = []
    for user, relevant in truth.items():
        recs = model.recommend(user, k)
        hits = len(set(recs) & relevant)
        p, r = hits/k, hits/len(relevant)
        dcg = sum(1/np.log2(j+2) for j, item in enumerate(recs) if item in relevant)
        idcg = sum(1/np.log2(j+2) for j in range(min(k, len(relevant))))
        rows.append([user, p, r, 2*p*r/(p+r) if p+r else 0, dcg/idcg])
    columns = ['user_id', f'Precision@{k}', f'Recall@{k}', f'F1@{k}', f'NDCG@{k}']
    details = pd.DataFrame(rows, columns=columns)
    metrics = details[columns[1:]].mean().to_dict()
    metrics.update(evaluated_users=len(details), heldout_rows=len(heldout),
                   evaluated_rows=len(known), dropped_rows=len(heldout)-len(known))
    return metrics, details


class FactorModel(Popularity):
    """Optional checkpoint exported from the user's original 04 / thu_nghiem.
    npz uses numeric arrays + strings only; pickle is never loaded.
    """
    def fit_checkpoint(self, train, path):
        super().fit(train)
        with np.load(path, allow_pickle=False) as artifact:
            self.kind = str(artifact['kind'].item())
            if self.kind not in ['bpr', 'mf']:
                raise ValueError('Loại checkpoint không hợp lệ.')
            if str(artifact['train_fingerprint'].item()) != fingerprint(train):
                raise ValueError('Checkpoint không khớp train hiện tại. Xuất lại sau khi train đúng dữ liệu.')
            if not np.array_equal(artifact['user_ids'], self.users) or not np.array_equal(artifact['movie_ids'], self.items):
                raise ValueError('ID mapping trong checkpoint không khớp train.')
            self.P = artifact['user_factors'].copy()
            self.Q = artifact['movie_factors'].copy()
            if self.P.ndim != 2 or self.Q.ndim != 2 or self.P.shape[0] != len(self.users) or self.Q.shape != (len(self.items), self.P.shape[1]):
                raise ValueError('Kích thước embedding sai.')
            if not np.isfinite(self.P).all() or not np.isfinite(self.Q).all():
                raise ValueError('Embedding chứa NaN/Inf.')
            if self.kind == 'mf':
                self.mean = float(artifact['global_mean'].item())
                self.bu, self.bi = artifact['user_bias'].copy(), artifact['movie_bias'].copy()
                if self.bu.shape != (len(self.users),) or self.bi.shape != (len(self.items),):
                    raise ValueError('Kích thước bias sai.')
                if not np.isfinite(self.mean) or not np.isfinite(self.bu).all() or not np.isfinite(self.bi).all():
                    raise ValueError('Bias chứa NaN/Inf.')
        return self

    def scores(self, user):
        if user not in self.user_pos:
            return self.pop.copy()
        u = self.user_pos[user]
        result = self.Q @ self.P[u]
        if self.kind == 'mf':
            result = result + self.mean + self.bu[u] + self.bi
        return result
