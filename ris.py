import numpy as np
import pandas as pd
import random
from collections import Counter, deque
import pickle
import os
import time
import re
from datetime import datetime

EPS = 1e-12

def _nodes_and_inmap(G: pd.DataFrame):
    nodes = np.unique(np.concatenate([G['source'].to_numpy(), G['target'].to_numpy()]))
    #print(nodes)
    in_map = G.groupby('target')['source'].apply(list).to_dict()
    return nodes, in_map

def _next_skip(p: float) -> int:
    if p <= 0.0:
        return 10**9
    if p >= 1.0 - EPS:
        return 0
    u = random.random()
    if u < EPS:
        u = EPS
    denom = np.log(1.0 - p)
    return int(np.floor(np.log(u) / denom))

def get_RRS_SUBSIM_WC_precomp(nodes, in_map):
    v = random.choice(list(nodes))
    R, activated = [v], {v}
    Q = deque([v])
    while Q:
        u = Q.popleft()
        nbrs = in_map.get(u, [])
        deg = len(nbrs)
        if deg == 0:
            continue
        p = 1.0 / deg
        i = _next_skip(p)
        while i < deg:
            w = nbrs[i]
            if w not in activated:
                activated.add(w)
                R.append(w)
                Q.append(w)
            i += _next_skip(p) + 1
    return R

def get_RRS_LT_live_edge(G: pd.DataFrame, weight_col: str | None = None):
    nodes = np.unique(np.concatenate([G['source'].to_numpy(), G['target'].to_numpy()]))
    v = random.choice(list(nodes))
    grouped = G.groupby('target')
    weights_map = {}
    for tgt, df in grouped:
        sources = df['source'].to_numpy().tolist()
        if weight_col and weight_col in df.columns:
            w = df[weight_col].to_numpy().astype(float)
        else:
            deg = len(sources)
            w = np.ones(deg, dtype=float) / deg if deg > 0 else np.array([], dtype=float)
        weights_map[tgt] = (sources, w)
    R, activated = [v], {v}
    Q = deque([v])
    while Q:
        u = Q.popleft()
        sources, w = weights_map.get(u, ([], np.array([], dtype=float)))
        if len(sources) == 0:
            continue
        sum_w = float(w.sum()) if w.size > 0 else 0.0
        if sum_w <= 0:
            continue
        if sum_w > 1 + 1e-12:
            w = w / sum_w
            sum_w = 1.0
        r = random.random()
        if r > sum_w:
            continue
        acc = 0.0
        parent = None
        for idx, wi in enumerate(w.tolist()):
            acc += wi
            if r <= acc:
                parent = sources[idx]
                break
        if parent is not None and parent not in activated:
            activated.add(parent)
            R.append(parent)
            Q.append(parent)
    return R

def solve_ris(seed, n, mc):
    res = [0] * n
    for i in seed:
        res[i[0]] = i[1] / mc
    return res

def ris_subsim(G: pd.DataFrame, n: int, mc: int = 1000):
    nodes, in_map = _nodes_and_inmap(G)
    R = []
    block_start = time.perf_counter()
    for t in range(1, mc + 1):
        R.append(get_RRS_SUBSIM_WC_precomp(nodes, in_map))
        if t % 100 == 0:
            elapsed = time.perf_counter() - block_start
            print(f"[SUBSIM] {t} rounds elapsed: {elapsed:.3f}s")
            block_start = time.perf_counter()
    return R

def select(R, k, SEED, G, mc):
    SEED = SEED.copy()
    for s in SEED:
        R = [rr for rr in R if s not in rr]
    if len(R) < mc:
        nodes, in_map = _nodes_and_inmap(G)
        for _ in range(mc - len(R)):
            R.append(get_RRS_SUBSIM_WC_precomp(nodes, in_map))
    for _ in range(k):
        if not R:
            break
        flat = [x for rr in R for x in rr]
        if not flat:
            break
        s = Counter(flat).most_common(1)[0][0]
        SEED.append(s)
        R = [rr for rr in R if s not in rr]
    return SEED

def _normalize_seed_ids(seeds):
    for s in seeds:
        if isinstance(s, (tuple, list, np.ndarray)):
            yield int(s[0])
        else:
            yield int(s)


def rr_coverage(seeds, R):
    seed_ids = set(_normalize_seed_ids(seeds))
    count = 0
    for rr in R:
        if not seed_ids.isdisjoint(rr):
            count += 1
    return count


def build_rr_index(R):
    idx = {}
    for i, rr in enumerate(R):
        for v in rr:
            if v in idx:
                idx[v].add(i)
            else:
                idx[v] = {i}
    return idx


def rr_coverage_with_index(seeds, rr_index):
    seed_ids = set(_normalize_seed_ids(seeds))
    covered = set()
    for v in seed_ids:
        covered.update(rr_index.get(v, ()))
    return len(covered)


def rr_stats(R):
    count = 0
    total = 0
    min_sz = 10**9
    max_sz = 0
    for rr in R:
        s = len(rr)
        count += 1
        total += s
        if s < min_sz:
            min_sz = s
        if s > max_sz:
            max_sz = s
    avg = (total / count) if count else 0.0
    if count == 0:
        min_sz = 0
    return {'count': count, 'avg': avg, 'min': min_sz, 'max': max_sz}

def make_ris(data_path, method='SUBSIM', mc=10000, runs=100):

    base_name = os.path.basename(data_path)
    dataset_key = base_name.split('_mean_')[0] if '_mean_' in base_name else os.path.splitext(base_name)[0]
    base_dir = os.path.dirname(data_path)

    with open(data_path, 'rb') as f:
        obj = pickle.load(f)

    if isinstance(obj, dict) and 'adj' in obj:
        adj = obj['adj']
    else:
        adj = obj

    try:
        src, tgt = adj.nonzero()
    except AttributeError:
        import scipy.sparse as sp
        if isinstance(adj, np.ndarray):
            adj = sp.csr_matrix(adj)
            src, tgt = adj.nonzero()
        else:
            raise TypeError(f"{os.path.basename(data_path)} must be SciPy sparse matrix or dict with 'adj'")

    d = pd.DataFrame({'source': src, 'target': tgt})

    mc = min(int(adj.shape[0]*0.1), mc) 
    print(f"[{dataset_key}] mc={mc}")

    if method == 'SUBSIM':
        R = ris_subsim(d, adj.shape[0], mc=mc)
    elif method in ('LT', 'SIS'):
        R = ris_subsim(d, adj.shape[0], mc=mc)
    else:
        raise ValueError(f"Unknown method '{method}'")

    m = re.search(r'_(?:mean_)?(IC|LT|SIS)(\d+)\.SG$', base_name, re.IGNORECASE)
    if m:
        rate_num = int(m.group(2))
        seed_frac_val = rate_num / 1000.0
        print(f"[{dataset_key}] Inferred seed_frac={seed_frac_val:.3f} from '{base_name}'")
    else:
        seed_frac_val = 0.05
        print(f"[{dataset_key}] No seed rate tag in filename. Using default seed_frac={seed_frac_val:.3f}")

    n_nodes = adj.shape[0]
    seed_size = max(1, int(seed_frac_val * n_nodes))
    seed_tensor = np.zeros((runs, n_nodes), dtype=np.int8)

    start_time = time.perf_counter()
    for run in range(runs):
        chosen_nodes = np.random.choice(np.arange(n_nodes), size=seed_size, replace=False)
        seed_tensor[run, chosen_nodes] = 1
    elapsed_seed = time.perf_counter() - start_time

    idx = build_rr_index(R)
    coverage_list = []
    start_time = time.perf_counter()
    for run in range(runs):
        seeds = np.where(seed_tensor[run] == 1)[0].tolist()
        cov = rr_coverage_with_index(seeds, idx)
        coverage_list.append(cov)
    elapsed_cov = time.perf_counter() - start_time

    stats = rr_stats(R)
    print(f"[{dataset_key}] RR stats: count={stats['count']}\nSingle node appearance times: avg={stats['avg']:.3f}, min={stats['min']}, max={stats['max']}\n")
    print('After sampling:')
    print(f"[{dataset_key}] {runs} runs, {int(seed_frac_val*100)}% seeds: mean coverage={np.mean(coverage_list):.3f}")
    print(f"[{dataset_key}] Seed gen {elapsed_seed:.3f}s, coverage {elapsed_cov:.3f}s, total {elapsed_seed+elapsed_cov:.3f}s")

    timestamp = datetime.now().strftime('%Y_%m_%d_%H%M')
    new_sg = {
        'adj': adj,
        'seed': seed_tensor,
        'coverage': coverage_list,
        'label': f'{dataset_key}_with_seed_{timestamp}',
        'source_file': data_path,
        'method': method,
    }
    if isinstance(obj, dict):
        for k in ('inverse_pairs', 'prob_matrix', 'attr_matrix', 'labels'):
            if k in obj and k not in new_sg:
                new_sg[k] = obj[k]

    out_dir = os.path.join(base_dir, dataset_key)
    os.makedirs(out_dir, exist_ok=True)
    diff_tag = m.group(1).upper() if m else 'SEED'
    rate_num_tag = int(m.group(2)) if m else int(round(seed_frac_val * 1000))
    out_path = os.path.join(out_dir, f'{dataset_key}_with_seed_{diff_tag}{rate_num_tag}_{timestamp}.SG')
    with open(out_path, 'wb') as f:
        pickle.dump(new_sg, f)
    print(f"[{dataset_key}] Saved SG with seed feature: {out_path}\n")
    return out_path

def make_digg_ris(seed_rate=0.05):
    import pickle
    import numpy as np
    import time
    import os
    from datetime import datetime
    try:
        with open('./data/digg/digg.SG', 'rb') as f:
            obj = pickle.load(f)
    except FileNotFoundError:
        raise FileNotFoundError("Missing ./data/digg/digg.SG. Run data/digg/build_digg_sg.py first.")

    if isinstance(obj, dict) and 'adj' in obj:
        adj = obj['adj']
    else:
        adj = obj

    try:
        src, tgt = adj.nonzero()
    except AttributeError:
        import scipy.sparse as sp
        if isinstance(adj, np.ndarray):
            adj = sp.csr_matrix(adj)
            src, tgt = adj.nonzero()
        else:
            raise TypeError("digg.SG must be a SciPy sparse matrix or dict with 'adj' matrix.")

    d = pd.DataFrame({'source': src, 'target': tgt}) 
    R = ris_subsim(d, adj.shape[0], mc=10000)
    stats = rr_stats(R)
    print(f"RR stats: set size={stats['count']}\nSingle node appearance times:avg={stats['avg']:.3f}, min={stats['min']}, max={stats['max']}")

    n_nodes = adj.shape[0]
    seed_size = max(1, int(seed_rate * n_nodes))
    seed_tensor = np.zeros((100, n_nodes), dtype=np.int8)

    start_time = time.perf_counter()
    for run in range(100):
        candidates = np.arange(n_nodes)
        chosen = np.random.choice(candidates, size=seed_size, replace=False)
        seed_tensor[run, chosen] = 1
    elapsed_seed = time.perf_counter() - start_time

    idx = build_rr_index(R)
    coverage_list = []
    start_time = time.perf_counter()
    for run in range(100):
        seeds = np.where(seed_tensor[run] == 1)[0].tolist()
        cov = rr_coverage_with_index(seeds, idx)
        coverage_list.append(cov)
    elapsed_cov = time.perf_counter() - start_time

    print("Coverage over 100 random 5% seed runs:")
    print("Average covered RR-set count:", np.mean(coverage_list))
    #print("Coverage distribution:", coverage_list[:10], "...")
    #print(f"Seed generation: {elapsed_seed:.3f}s, coverage computation: {elapsed_cov:.3f}s, total: {elapsed_seed + elapsed_cov:.3f}s")
    
    timestamp = datetime.now().strftime('%Y_%m_%d_%H%M')
    dataset_name = 'digg'
    chosen_path = './data/digg/digg.SG'
    method = 'SUBSIM'
    new_sg = {
        'adj': adj,
        'seed': seed_tensor,
        'coverage': coverage_list,
        'label': f'{dataset_name}_{timestamp}',
        'source_file': chosen_path,
        'method': method,
    }

    out_dir = './data/digg/digg'
    os.makedirs(out_dir, exist_ok=True)
    save_path = os.path.join(out_dir, f'{dataset_name}_with_seed_SEED{seed_rate}_{timestamp}.SG')
    with open(save_path, 'wb') as f:
        pickle.dump(new_sg, f)

    print(f"Saved SG file with seed feature: {save_path}")

if __name__ == '__main__':
    make_ris('data/cora_ml_mean_IC50.SG')
    make_digg_ris()