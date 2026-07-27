#!/usr/bin/env python3
"""
MAXCUT QAOA — 20 Nodes: Valley Search vs Layer-by-Layer (LBL)
================================================================
Hilbert space: 2^20 = 1,048,576 states — exact simulation is cheap.
Goal: compare Valley's "scan geometry → descend" with LBL's
"optimize layer-by-layer" on a sparse random 20-node graph.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import time, argparse, sys
from collections import defaultdict

# ============================================================
# 0.  Graph generation
# ============================================================
def generate_graph_20(edge_prob=0.22, seed=42):
    """Generate a connected 20-node Erdos-Renyi graph with non-trivial degree spread."""
    rng = np.random.default_rng(seed)
    while True:
        adj = np.zeros((20, 20), dtype=np.int8)
        for i in range(20):
            for j in range(i + 1, 20):
                if rng.random() < edge_prob:
                    adj[i, j] = adj[j, i] = 1
        # connectivity check
        visited = np.zeros(20, dtype=bool)
        stack = [0]; visited[0] = True
        while stack:
            u = stack.pop()
            for v in range(20):
                if adj[u, v] and not visited[v]:
                    visited[v] = True; stack.append(v)
        if not visited.all():
            continue
        deg = adj.sum(axis=1)
        if len(set(deg)) >= 4 and deg.min() >= 1 and deg.max() <= 12:
            return adj

def draw_graph(adj, title="20-Node Random Graph", save_path="graph_20node.png"):
    """Draw the graph with a circular layout — clean & readable."""
    import networkx as nx
    G = nx.from_numpy_array(adj.astype(int))
    n = len(adj)
    edges_list = [(i, j) for i in range(n) for j in range(i+1, n) if adj[i,j]]

    # circular layout
    pos = {}
    for i in range(n):
        angle = 2 * np.pi * i / n - np.pi / 2
        pos[i] = (np.cos(angle), np.sin(angle))

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.set_xlim(-1.35, 1.35)
    ax.set_ylim(-1.35, 1.35)
    ax.set_aspect('equal')
    ax.axis('off')

    # edges
    for u, v in edges_list:
        x1, y1 = pos[u]; x2, y2 = pos[v]
        ax.plot([x1, x2], [y1, y2], color='#b0b0b0', lw=0.8, alpha=0.7, zorder=1)

    # nodes
    node_radius = 0.10
    degrees = adj.sum(axis=1).astype(int)
    max_deg = max(degrees)
    # color map: low degree = blue, high degree = red
    cmap = plt.cm.coolwarm
    for i in range(n):
        x, y = pos[i]
        d = degrees[i]
        color = cmap(0.15 + 0.7 * d / max_deg) if max_deg > 0 else cmap(0.5)
        circle = plt.Circle((x, y), node_radius, facecolor=color, edgecolor='#333333',
                            lw=1.5, zorder=3, alpha=0.92)
        ax.add_patch(circle)
        ax.text(x, y, str(i), ha='center', va='center', fontsize=10,
                fontweight='bold', color='white' if d > max_deg/2 else '#222222', zorder=4)

    # title & stats
    n_edges = len(edges_list)
    deg_str = f"deg ∈ [{degrees.min()}, {degrees.max()}]"
    ax.set_title(f"{title}\n20 nodes, {n_edges} edges, {deg_str}",
                 fontsize=14, fontweight='bold', pad=12)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Graph saved → {save_path}")
    return save_path


# ============================================================
# 1.  QAOA Simulator (exact, 20 qubits → 1M states, trivial)
# ============================================================
class QAOASimulator20:
    def __init__(self, adj):
        self.n = len(adj)
        self.dim = 1 << self.n
        self.edges = [(i, j) for i in range(self.n)
                      for j in range(i+1, self.n) if adj[i, j]]
        self.n_edges = len(self.edges)
        self.initial_state = np.ones(self.dim, dtype=np.complex64) / np.sqrt(self.dim)
        self._idx = np.arange(self.dim, dtype=np.uint32)

    def evolve(self, gammas, betas):
        state = self.initial_state.copy()
        for layer in range(len(gammas)):
            gamma, beta = float(gammas[layer]), float(betas[layer])
            # ZZ
            for i, j in self.edges:
                same = ((self._idx >> i) & 1) == ((self._idx >> j) & 1)
                phases = np.where(same, np.exp(-1j * gamma), np.exp(1j * gamma))
                state *= phases.astype(np.complex64)
            # RX — stride-based
            c, s = np.cos(beta), np.sin(beta)
            rx00, rx01 = np.complex64(c), np.complex64(-1j * s)
            rx10, rx11 = np.complex64(-1j * s), np.complex64(c)
            for qi in range(self.n):
                step = 1 << (qi + 1)
                half = 1 << qi
                for start in range(0, self.dim, step):
                    mid = start + half
                    end = start + step
                    a0 = state[start:mid].copy()
                    a1 = state[mid:end].copy()
                    state[start:mid] = rx00 * a0 + rx01 * a1
                    state[mid:end]   = rx10 * a0 + rx11 * a1
        return state

    def cost(self, gammas, betas):
        state = self.evolve(gammas, betas)
        probs = (state.real.astype(np.float32)**2
                 + state.imag.astype(np.float32)**2)
        total = np.float32(0.0)
        for i, j in self.edges:
            same = ((self._idx >> i) & 1) == ((self._idx >> j) & 1)
            total += 2.0 * probs[same].sum() - 1.0
        return float(total)

    def maxcut_from_cost(self, gammas, betas):
        c = self.cost(gammas, betas)
        return (self.n_edges - c) / 2.0

    def cost_gradient(self, gammas, betas):
        """Numerical gradient via symmetric finite diff (h=0.01)."""
        p = len(gammas)
        grad_g = np.zeros(p, dtype=np.float64)
        grad_b = np.zeros(p, dtype=np.float64)
        h = 0.01
        g = np.array(gammas, dtype=np.float64)
        b = np.array(betas, dtype=np.float64)
        for k in range(p):
            g_plus = g.copy();  g_plus[k] += h
            g_minus = g.copy(); g_minus[k] -= h
            grad_g[k] = (self.cost(g_plus, b) - self.cost(g_minus, b)) / (2 * h)
            b_plus = b.copy();  b_plus[k] += h
            b_minus = b.copy(); b_minus[k] -= h
            grad_b[k] = (self.cost(g, b_plus) - self.cost(g, b_minus)) / (2 * h)
        return grad_g, grad_b


# ============================================================
# 2.  Exact MaxCut  (2^20 = 1M — fast enumeration)
# ============================================================
def exact_maxcut(adj):
    n = len(adj)
    edges_list = [(i, j) for i in range(n)
                  for j in range(i+1, n) if adj[i, j]]
    n_edges = len(edges_list)
    edge_bits = [(1 << i, 1 << j) for i, j in edges_list]
    dim = 1 << n
    batch = 1 << 19  # 512K per batch
    best_cut = -1
    for start in range(0, dim, batch):
        end = min(start + batch, dim)
        idx = np.arange(start, end, dtype=np.uint32)
        cuts = np.zeros(len(idx), dtype=np.int32)
        for mi, mj in edge_bits:
            cuts += ((idx & mi) != 0) != ((idx & mj) != 0)
        b = int(np.max(cuts))
        if b > best_cut:
            best_cut = b
    return best_cut, n_edges


# ============================================================
# 3.  Valley Search  (p=2, full landscape-aware)
# ============================================================
def valley_search(sim, n_layers=2, n_coarse=3, n_top=6, n_refine=6,
                  n_iter=50, n_scan_dir=20, n_scan_samp=30):
    """
    Full Valley Search:
      1) Coarse grid scan → pick top candidates
      2) Projection variance scan → detect narrow valleys
      3) Refine top points → Adam from best starting positions
    """
    results = {'history': [], 'best_params': None, 'best_cut': -np.inf,
               'n_cost_evals': 0, 'valley_score': None, 'time': 0}
    t0 = time.time()

    # --- 3a. Coarse grid (hypercube)
    dims = 2 * n_layers  # gamma_1,beta_1,...,gamma_p,beta_p
    grid = np.linspace(0, 2 * np.pi, n_coarse + 2)[1:-1]  # skip 0 & 2pi
    mesh = np.meshgrid(*[grid] * dims, indexing='ij')
    flat = np.stack([m.ravel() for m in mesh], axis=1)
    cuts = np.zeros(len(flat))
    for idx, pt in enumerate(flat):
        g = pt[0::2]; b = pt[1::2]
        cuts[idx] = sim.maxcut_from_cost(g, b)
        if (idx + 1) % max(1, len(flat) // 4) == 0:
            print(f"      coarse {idx+1}/{len(flat)} best={cuts[:idx+1].max():.2f}", end='\r', flush=True)
    results['n_cost_evals'] += len(flat)
    top_idx = np.argsort(cuts)[-n_top:]
    top_pts = flat[top_idx]
    print(f"\n      Coarse done: {len(flat)} pts, best={cuts[top_idx[-1]]:.2f}")

    # --- 3b. Projection variance → detect valleys
    print(f"    [Valley Detection] scanning {n_scan_dir} directions × {n_scan_samp} samples...")
    best_pt = top_pts[-1].copy()
    variances = []
    for d in range(n_scan_dir):
        # random normalized direction
        direction = np.random.randn(dims)
        direction /= np.linalg.norm(direction) + 1e-12
        samples = np.linspace(-np.pi / 2, np.pi / 2, n_scan_samp)
        vals = np.zeros(n_scan_samp)
        for si, s in enumerate(samples):
            pt = best_pt + s * direction
            g = pt[0::2]; b = pt[1::2]
            vals[si] = sim.maxcut_from_cost(g, b)
        results['n_cost_evals'] += n_scan_samp
        variances.append(float(np.var(vals)))
    valley_score = float(np.max(variances)) if variances else 0.0
    results['valley_score'] = valley_score
    results['all_variances'] = variances
    print(f"      Variance range: [{min(variances):.4f}, {max(variances):.4f}]  score={valley_score:.4f}")

    # --- 3c. Refine from each top point (gradient-free Adam variant)
    # Reuse top_pts as starting points; refine locally.
    refine_best = -np.inf
    refine_best_params = None
    for pi, pt0 in enumerate(top_pts):
        current = pt0.copy()
        m = np.zeros(dims); v = np.zeros(dims)
        beta1, beta2, eps_adam = 0.9, 0.999, 1e-8
        lr = 0.05
        best_local_cut = -np.inf
        best_local_pt = current.copy()
        for t in range(1, n_refine + 1):
            g = current[0::2]; b = current[1::2]
            grad_g, grad_b = sim.cost_gradient(g, b)
            results['n_cost_evals'] += 2 * n_layers  # 2*grad calls (central diff)
            # for maxcut we ascend, not descend
            gr = np.zeros(dims)
            gr[0::2] = grad_g; gr[1::2] = grad_b
            m = beta1 * m + (1 - beta1) * gr
            v = beta2 * v + (1 - beta2) * (gr ** 2)
            m_hat = m / (1 - beta1 ** t)
            v_hat = v / (1 - beta2 ** t)
            current += lr * m_hat / (np.sqrt(v_hat) + eps_adam)
            # evaluate
            cut_val = sim.maxcut_from_cost(current[0::2], current[1::2])
            results['n_cost_evals'] += 1
            results['history'].append((pi, t, cut_val))
            if cut_val > best_local_cut:
                best_local_cut = cut_val
                best_local_pt = current.copy()
            if t % max(1, n_refine // 4) == 0:
                print(f"        refine[{pi}] iter{t}: {cut_val:.3f}", end='\r', flush=True)
        if best_local_cut > refine_best:
            refine_best = best_local_cut
            refine_best_params = best_local_pt.copy()
        print(f"      refine[{pi}] final: {best_local_cut:.3f}")

    # --- Adam polishing from best refined point
    print(f"    [Adam Polish] from best refined ({refine_best:.3f})")
    current = refine_best_params.copy()
    m = np.zeros(dims); v = np.zeros(dims)
    best_cut = refine_best
    best_params = current.copy()
    for t in range(1, n_iter + 1):
        g = current[0::2]; b = current[1::2]
        grad_g, grad_b = sim.cost_gradient(g, b)
        results['n_cost_evals'] += 2 * n_layers
        gr = np.zeros(dims)
        gr[0::2] = grad_g; gr[1::2] = grad_b
        m = beta1 * m + (1 - beta1) * gr
        v = beta2 * v + (1 - beta2) * (gr ** 2)
        m_hat = m / (1 - beta1 ** t)
        v_hat = v / (1 - beta2 ** t)
        current += lr * m_hat / (np.sqrt(v_hat) + eps_adam)
        cut_val = sim.maxcut_from_cost(current[0::2], current[1::2])
        results['n_cost_evals'] += 1
        results['history'].append((-1, t, cut_val))
        if cut_val > best_cut:
            best_cut = cut_val
            best_params = current.copy()
        if t % max(1, n_iter // 5) == 0:
            print(f"        adam {t:3d}: best={best_cut:.3f} current={cut_val:.3f}", end='\r', flush=True)
    print(f"\n      Adam done: best={best_cut:.3f}")
    results['best_cut'] = best_cut
    results['best_params'] = best_params.copy()
    results['time'] = time.time() - t0
    return results


# ============================================================
# 4.  Layer-by-Layer (LBL)
# ============================================================
def lbl_train(sim, n_layers=3, n_restarts=3, n_iter=40, lr=0.08):
    """
    Build QAOA layer by layer:
      1. Train p=1 to convergence from multiple restarts → fix (γ₁,β₁)
      2. Add layer 2, train only new params → fix (γ₂,β₂)
      3. Add layer 3, train only new params → fix (γ₃,β₃)
      (Optional full polish at the end.)
    """
    results = {'history': [], 'best_params': None, 'best_cut': -np.inf,
               'n_cost_evals': 0, 'time': 0, 'layer_cuts': []}
    t0 = time.time()
    betas_opt = []; gammas_opt = []
    all_params = np.array([], dtype=np.float64)

    for layer in range(1, n_layers + 1):
        layer_best_cut = -np.inf
        layer_best_params = None
        for restart in range(n_restarts):
            new_g = np.random.uniform(0.1, 2 * np.pi - 0.1)
            new_b = np.random.uniform(0.1, 2 * np.pi - 0.1)
            params = np.array([new_g, new_b], dtype=np.float64)  # only new layer
            m = np.zeros(2); v = np.zeros(2)
            beta1, beta2, eps = 0.9, 0.999, 1e-8
            for t in range(1, n_iter + 1):
                # full params (old fixed + new)
                full_g = np.append(gammas_opt, params[0])
                full_b = np.append(betas_opt, params[1])
                grad_g, grad_b = sim.cost_gradient(full_g, full_b)
                results['n_cost_evals'] += 2 * layer
                gr = np.array([grad_g[-1], grad_b[-1]])
                m = beta1 * m + (1 - beta1) * gr
                v = beta2 * v + (1 - beta2) * (gr ** 2)
                m_hat = m / (1 - beta1 ** t)
                v_hat = v / (1 - beta2 ** t)
                params += lr * m_hat / (np.sqrt(v_hat) + eps)
                # eval
                full_g_cur = np.append(gammas_opt, params[0])
                full_b_cur = np.append(betas_opt, params[1])
                cut_v = sim.maxcut_from_cost(full_g_cur, full_b_cur)
                results['n_cost_evals'] += 1
                results['history'].append((layer, restart, t, cut_v))
            # end of this restart
            full_g_f = np.append(gammas_opt, params[0])
            full_b_f = np.append(betas_opt, params[1])
            final_cut = sim.maxcut_from_cost(full_g_f, full_b_f)
            results['n_cost_evals'] += 1
            if final_cut > layer_best_cut:
                layer_best_cut = final_cut
                layer_best_params = params.copy()
            print(f"      LBL p={layer} restart {restart+1}: {final_cut:.3f}", end='\r', flush=True)
        # fix this layer
        gammas_opt = np.append(gammas_opt, layer_best_params[0])
        betas_opt  = np.append(betas_opt, layer_best_params[1])
        results['layer_cuts'].append(layer_best_cut)
        print(f"      LBL p={layer} done: best={layer_best_cut:.3f}")

    results['best_cut'] = layer_best_cut
    results['best_params'] = (gammas_opt.copy(), betas_opt.copy())
    results['time'] = time.time() - t0
    return results


# ============================================================
# 5.  LBL with full p=2 polish (tight comparison)
# ============================================================
def lbl_with_polish(sim, n_restarts=3, n_iter_lbl=40, n_iter_polish=50, lr=0.08):
    """LBL p=1 → p=2, then full Adam polish on all 4 params to be fair vs Valley p=2."""
    results = {'history': [], 'best_params': None, 'best_cut': -np.inf,
               'n_cost_evals': 0, 'time': 0, 'layer_cuts': []}
    t0 = time.time()

    # --- LBL: p=1
    g_opt, b_opt = [], []
    for layer in range(1, 3):  # p=1 then p=2
        layer_best_cut = -np.inf
        layer_best = None
        for rst in range(n_restarts):
            ng = np.random.uniform(0.1, 2*np.pi-0.1)
            nb = np.random.uniform(0.1, 2*np.pi-0.1)
            params = np.array([ng, nb]); m=np.zeros(2); v=np.zeros(2)
            for t in range(1, n_iter_lbl+1):
                fg = np.append(g_opt, params[0]); fb = np.append(b_opt, params[1])
                grad_g, grad_b = sim.cost_gradient(fg, fb)
                results['n_cost_evals'] += 2 * layer
                gr = np.array([grad_g[-1], grad_b[-1]])
                m = 0.9*m + 0.1*gr; v = 0.999*v + 0.001*(gr**2)
                m_h = m/(1-0.9**t); v_h = v/(1-0.999**t)
                params += lr * m_h / (np.sqrt(v_h)+1e-8)
                fg2=np.append(g_opt,params[0]); fb2=np.append(b_opt,params[1])
                cv=sim.maxcut_from_cost(fg2,fb2); results['n_cost_evals']+=1
                results['history'].append((layer,rst,t,cv))
            fgf=np.append(g_opt,params[0]); fbf=np.append(b_opt,params[1])
            fc=sim.maxcut_from_cost(fgf,fbf); results['n_cost_evals']+=1
            if fc>layer_best_cut: layer_best_cut=fc; layer_best=params.copy()
        g_opt=np.append(g_opt,layer_best[0]); b_opt=np.append(b_opt,layer_best[1])
        results['layer_cuts'].append(layer_best_cut)
        print(f"      LBL p={layer}: {layer_best_cut:.3f}")

    # --- Full polish (p=2, 4 params)
    print(f"    [LBL Polish] p=2 full Adam ({n_iter_polish} iters)")
    params = np.zeros(4); params[0::2]=g_opt; params[1::2]=b_opt
    m=np.zeros(4); v=np.zeros(4); best_cut=-np.inf; best_p=params.copy()
    for t in range(1, n_iter_polish+1):
        g=params[0::2]; b=params[1::2]
        grad_g,grad_b=sim.cost_gradient(g,b); results['n_cost_evals']+=4
        gr=np.zeros(4); gr[0::2]=grad_g; gr[1::2]=grad_b
        m=0.9*m+0.1*gr; v=0.999*v+0.001*(gr**2)
        m_h=m/(1-0.9**t); v_h=v/(1-0.999**t)
        params+=lr*m_h/(np.sqrt(v_h)+1e-8)
        cv=sim.maxcut_from_cost(params[0::2],params[1::2]); results['n_cost_evals']+=1
        results['history'].append((3,0,t,cv))
        if cv>best_cut: best_cut=cv; best_p=params.copy()
        if t%max(1,n_iter_polish//5)==0:
            print(f"        polish {t:3d}: best={best_cut:.3f}", end='\r',flush=True)
    print(f"\n      Polish done: {best_cut:.3f}")
    results['best_cut']=best_cut
    results['best_params']=(best_p[0::2].copy(),best_p[1::2].copy())
    results['time']=time.time()-t0
    return results


# ============================================================
# 6.  Visualization
# ============================================================
def plot_valley_detection(variances, prefix='20node'):
    """Histogram of projection variances → valley depth distribution."""
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(variances, bins=15, color='#5D9CEC', edgecolor='white', alpha=0.85)
    ax.axvline(np.mean(variances), color='#E74C3C', ls='--', lw=2,
               label=f'mean = {np.mean(variances):.4f}')
    ax.set_xlabel('Projection Variance (narrowness of valley)', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Valley Detection — Projection Variance Distribution', fontsize=14, fontweight='bold')
    ax.legend()
    fig.tight_layout()
    fig.savefig(f'valley_detection_{prefix}.png', dpi=150)
    plt.close(fig)

def plot_coarse_vs_refined(valley_results, exact_cut, prefix='20node'):
    """Scatter: coarse evaluations → refined evaluations."""
    hist = valley_results['history']
    if not hist:
        print("    [skip] no valley history to plot")
        return
    refined = [(t, v) for _, t, v in hist if t > 0]
    fig, ax = plt.subplots(figsize=(8, 5))
    steps = [r[0] for r in refined]
    vals = [r[1] for r in refined]
    ax.plot(steps, vals, 'o-', color='#5D9CEC', ms=4, alpha=0.7)
    ax.axhline(exact_cut, color='#2ECC71', ls='--', lw=2, label=f'Exact MaxCut = {exact_cut}')
    ax.set_xlabel('Adam iteration', fontsize=12)
    ax.set_ylabel('Cut Value', fontsize=12)
    ax.set_title('Valley Search — Refinement Progress', fontsize=14, fontweight='bold')
    ax.legend()
    fig.tight_layout()
    fig.savefig(f'valley_refine_{prefix}.png', dpi=150)
    plt.close(fig)

def plot_lbl_layers(lbl_history, prefix='20node'):
    """Show LBL p=1→p=2→p=3 convergence by layer."""
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = {1: '#E74C3C', 2: '#F39C12', 3: '#2ECC71'}
    for layer, rst, t, val in lbl_history:
        ax.scatter(t + (layer-1)*100, val, color=colors[layer], s=8, alpha=0.5,
                   label=f'p={layer}' if rst==0 and t==1 else "")

    from collections import Counter
    handles_labels = {}
    for layer, rst, t, val in lbl_history:
        if f'p={layer}' not in handles_labels:
            handles_labels[f'p={layer}'] = colors[layer]
    for label, col in handles_labels.items():
        ax.scatter([], [], color=col, s=30, label=label)
    ax.legend()
    ax.set_xlabel('Training step', fontsize=12)
    ax.set_ylabel('Cut Value', fontsize=12)
    ax.set_title('LBL Training — Layer-by-Layer Convergence', fontsize=14, fontweight='bold')
    fig.tight_layout()
    fig.savefig(f'lbl_process_{prefix}.png', dpi=150)
    plt.close(fig)

def plot_head_to_head(valley_cut, lbl_cut, exact_cut, n_edges, valley_time, lbl_time,
                      valley_cost, lbl_cost, valley_score, prefix='20node'):
    """Side-by-side bar chart: Valley vs LBL vs Exact."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # --- left: cut values
    methods = ['Exact', 'Valley (p=2)', 'LBL (p=2)']
    cuts = [exact_cut, valley_cut, lbl_cut]
    ratios = [100, valley_cut/exact_cut*100, lbl_cut/exact_cut*100]
    bars = ax1.bar(methods, cuts, color=['#34495E', '#5D9CEC', '#F39C12'],
                   edgecolor='white', lw=1.5, width=0.55)
    ax1.set_ylim(0, n_edges * 1.15)
    # value labels
    for bar, val, r in zip(bars, cuts, ratios):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                 f'{val:.2f}\n({r:.1f}%)', ha='center', va='bottom', fontsize=11, fontweight='bold')
    ax1.set_title('Cut Values (20 nodes)', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Cut / Edges', fontsize=12)

    # --- right: stats table
    ax2.axis('off')
    table_data = [
        ['Metric', 'Valley (p=2)', 'LBL (p=2)'],
        ['Best Cut', f'{valley_cut:.3f}', f'{lbl_cut:.3f}'],
        ['Approx. Ratio', f'{valley_cut/exact_cut*100:.2f}%', f'{lbl_cut/exact_cut*100:.2f}%'],
        ['Runtime (s)', f'{valley_time:.1f}', f'{lbl_time:.1f}'],
        ['Cost Evals', str(valley_cost), str(lbl_cost)],
        ['Valley Score', f'{valley_score:.4f}', 'N/A'],
    ]
    table = ax2.table(cellText=table_data, cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.0, 1.5)
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor('#34495E')
            cell.set_text_props(color='white', fontweight='bold')
        elif col == 0:
            cell.set_facecolor('#ECF0F1')
            cell.set_text_props(fontweight='bold')
        elif row == 2:  # winner highlight
            pass
    ax2.set_title('Performance Comparison', fontsize=14, fontweight='bold')

    # winner text
    winner = 'Valley' if valley_cut >= lbl_cut else 'LBL'
    delta = abs(valley_cut - lbl_cut)
    fig.suptitle(f'Head-to-Head: winner = {winner} (Δ = {delta:.3f})',
                 fontsize=15, fontweight='bold', y=1.01)

    fig.tight_layout()
    fig.savefig(f'head_to_head_{prefix}.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Charts saved: head_to_head_{prefix}.png")


# ============================================================
# 7.  p=2 landscape slices
# ============================================================
def plot_p2_slices(sim, valley_params, lbl_params, exact_cut, prefix='20node'):
    """Plot 2D slices of the p=2 landscape centered at both optima."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    n_grid = 30
    delta = np.pi / 2

    for ax_idx, (label, params) in enumerate([('Valley', valley_params), ('LBL', lbl_params)]):
        ax = axes[ax_idx]
        g0, g1, b0, b1 = params[0], params[1], params[2], params[3]
        # fix beta, vary gamma
        Z = np.zeros((n_grid, n_grid))
        g_range = np.linspace(g0 - delta, g0 + delta, n_grid) if ax_idx == 0 else \
                  np.linspace(g1 - delta, g1 + delta, n_grid)
        b_range = np.linspace(b0 - delta, b0 + delta, n_grid) if ax_idx == 0 else \
                  np.linspace(b1 - delta, b1 + delta, n_grid)
        # We'll do gamma1 vs gamma2 slice and beta1 vs beta2 slice separately.
        # Simpler: gamma1 vs beta1 slice around [γ₁,β₁] at fixed [γ₂,β₂]
        if ax_idx == 0:
            gv = np.linspace(g0 - delta, g0 + delta, n_grid)
            bv = np.linspace(b0 - delta, b0 + delta, n_grid)
        else:
            gv = np.linspace(-delta/2, delta/2, n_grid)
            bv = np.linspace(-delta/2, delta/2, n_grid)

        for ig, _g in enumerate(gv):
            for ib, _b in enumerate(bv):
                if ax_idx == 0:
                    Z[ig, ib] = sim.maxcut_from_cost([_g, g1], [_b, b1])
                else:
                    Z[ig, ib] = sim.maxcut_from_cost([_g + lbl_params[0], lbl_params[2]],
                                                     [_b + lbl_params[1], lbl_params[3]])

        im = ax.contourf(gv, bv, Z, levels=20, cmap='RdYlBu')
        ax.scatter([params[0]], [params[1]], color='black', s=80, marker='*',
                   zorder=5, edgecolors='white', lw=1)
        ax.set_xlabel('γ₁' if ax_idx == 0 else 'γ₁ (rel)', fontsize=11)
        ax.set_ylabel('β₁' if ax_idx == 0 else 'β₁ (rel)', fontsize=11)
        ax.set_title(f'{label} p=2 Landscape (γ₁,β₁ slice)', fontsize=12, fontweight='bold')
        plt.colorbar(im, ax=ax, shrink=0.8)

    fig.suptitle(f'p=2 QAOA Landscape Slices — Exact MaxCut = {exact_cut}', fontsize=14, fontweight='bold')
    fig.tight_layout()
    fig.savefig(f'landscape_p2_{prefix}.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Landscape chart saved: landscape_p2_{prefix}.png")


# ============================================================
# 8.  Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(description='20-Node MAXCUT: Valley vs LBL')
    parser.add_argument('--fast', action='store_true', help='Quick mode')
    parser.add_argument('--full', action='store_true', help='Full comparison')
    parser.add_argument('--p', type=int, default=2, help='QAOA depth (default 2)')
    args = parser.parse_args()

    # Config
    if args.fast:
        cfg = {
            'n_coarse': 2, 'n_top': 4, 'n_refine': 4,
            'valley_iter': 30, 'valley_scan_dir': 10, 'valley_scan_samp': 20,
            'lbl_restarts': 2, 'lbl_iter': 25, 'lbl_polish': 30, 'lbl_n_layers': 3,
            'description': 'Fast (~15-20 min)'
        }
    elif args.full:
        cfg = {
            'n_coarse': 3, 'n_top': 8, 'n_refine': 8,
            'valley_iter': 80, 'valley_scan_dir': 25, 'valley_scan_samp': 35,
            'lbl_restarts': 4, 'lbl_iter': 50, 'lbl_polish': 60, 'lbl_n_layers': 3,
            'description': 'Full (~2-3 hours)'
        }
    else:
        cfg = {
            'n_coarse': 2, 'n_top': 5, 'n_refine': 6,
            'valley_iter': 50, 'valley_scan_dir': 15, 'valley_scan_samp': 25,
            'lbl_restarts': 3, 'lbl_iter': 35, 'lbl_polish': 40, 'lbl_n_layers': 3,
            'description': 'Balanced (~40-60 min)'
        }

    p = args.p

    print("=" * 70)
    print(f"  MAXCUT QAOA — 20 NODES: Valley vs LBL (p={p})")
    print(f"  Mode: {cfg['description']}")
    print(f"  Hilbert space: 2^20 = 1,048,576 states")
    print("=" * 70)

    # ---- 1. Generate graph ----
    print("\n[1] Generating 20-node graph...")
    adj = generate_graph_20(edge_prob=0.22, seed=42)
    n_edges = int(adj.sum() // 2)
    degrees = sorted(int(d) for d in adj.sum(axis=1))
    print(f"    Graph: 20 nodes, {n_edges} edges, deg ∈ [{degrees[0]}, {degrees[-1]}]")

    # ---- 2. Exact MaxCut ----
    print("\n[2] Exact MaxCut...")
    t0 = time.time()
    exact_cut, n_edges = exact_maxcut(adj)
    print(f"    Exact MaxCut = {exact_cut}/{n_edges} = {exact_cut/n_edges*100:.1f}%  ({time.time()-t0:.1f}s)")

    # ---- 3. Init simulator ----
    print("\n[3] Initializing QAOA simulator...")
    sim = QAOASimulator20(adj)
    print(f"    dim = {sim.dim:,}, edges = {sim.n_edges}")
    # quick sanity
    c_sanity = sim.maxcut_from_cost([1.0], [0.5])
    print(f"    sanity check p=1: cut = {c_sanity:.3f}")

    # ---- 4. Valley Search ----
    print(f"\n[4] Valley Search (p={p})...")
    valley = valley_search(sim, n_layers=p,
                           n_coarse=cfg['n_coarse'],
                           n_top=cfg['n_top'],
                           n_refine=cfg['n_refine'],
                           n_iter=cfg['valley_iter'],
                           n_scan_dir=cfg['valley_scan_dir'],
                           n_scan_samp=cfg['valley_scan_samp'])
    print(f"    Valley best: {valley['best_cut']:.3f} = {valley['best_cut']/exact_cut*100:.2f}%")
    print(f"    Valley time: {valley['time']:.1f}s, cost evals: {valley['n_cost_evals']}")
    print(f"    Valley score: {valley['valley_score']:.4f}")

    # ---- 5. LBL Search ----
    print(f"\n[5] LBL Search (p=1→{p}, polish)...")
    lbl = lbl_with_polish(sim,
                          n_restarts=cfg['lbl_restarts'],
                          n_iter_lbl=cfg['lbl_iter'],
                          n_iter_polish=cfg['lbl_polish'],
                          lr=0.08)
    print(f"    LBL best: {lbl['best_cut']:.3f} = {lbl['best_cut']/exact_cut*100:.2f}%")
    print(f"    LBL time: {lbl['time']:.1f}s, cost evals: {lbl['n_cost_evals']}")

    # ---- 6. p=1 landscape (quick)
    print(f"\n[6] p=1 QAOA landscape (25×25)...")
    n_g = 25
    gv = np.linspace(0, 2*np.pi, n_g)
    Z1 = np.zeros((n_g, n_g))
    for i, g in enumerate(gv):
        for j, b in enumerate(gv):
            Z1[i, j] = sim.maxcut_from_cost([g], [b])
    p1_best = float(np.max(Z1))
    print(f"    p=1 best: {p1_best:.3f} ({p1_best/exact_cut*100:.1f}%)")

    # ---- 7. Visualizations ----
    print(f"\n[7] Generating charts...")

    # valley detection
    if valley.get('all_variances'):
        plot_valley_detection(valley['all_variances'])

    # valley refinement
    plot_coarse_vs_refined(valley, exact_cut)

    # LBL convergence
    plot_lbl_layers(lbl['history'])

    # p=2 landscape slices
    vp = valley['best_params']
    lp = lbl['best_params']
    if len(vp) == 4 and len(lp[0]) == 2:
        plot_p2_slices(sim, vp,
                       np.array([lp[0][0], lp[1][0], lp[0][1], lp[1][1]]),
                       exact_cut)

    # Head-to-head
    plot_head_to_head(
        valley['best_cut'], lbl['best_cut'], exact_cut, n_edges,
        valley['time'], lbl['time'],
        valley['n_cost_evals'], lbl['n_cost_evals'],
        valley.get('valley_score', 0.0)
    )

    # p=1 landscape
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.contourf(gv, gv, Z1, levels=25, cmap='RdYlBu')
    ax.set_xlabel('γ', fontsize=13)
    ax.set_ylabel('β', fontsize=13)
    ax.set_title(f'p=1 QAOA Landscape (best cut = {p1_best:.3f})', fontsize=14, fontweight='bold')
    plt.colorbar(im, ax=ax)
    # mark best
    i_max, j_max = np.unravel_index(np.argmax(Z1), Z1.shape)
    ax.scatter([gv[i_max]], [gv[j_max]], color='black', s=80, marker='*',
               edgecolors='white', lw=1.5, zorder=5)
    fig.tight_layout()
    fig.savefig('landscape_p1_20node.png', dpi=150, bbox_inches='tight')
    plt.close(fig)

    # ---- Summary ----
    print("\n" + "=" * 70)
    print("  RESULTS SUMMARY")
    print("=" * 70)
    print(f"  Graph:         20 nodes, {n_edges} edges")
    print(f"  Exact MaxCut:  {exact_cut}/{n_edges} = {exact_cut/n_edges*100:.1f}%")
    print(f"  p=1 best:      {p1_best:.3f} ({p1_best/exact_cut*100:.1f}%)")
    print(f"  ───────────────────────────────")
    print(f"  Valley (p={p}):  {valley['best_cut']:.3f}  ({valley['best_cut']/exact_cut*100:.2f}%)")
    print(f"                   time: {valley['time']:.1f}s  evals: {valley['n_cost_evals']}")
    print(f"                   valley score: {valley.get('valley_score',0):.4f}")
    print(f"  LBL (p={p}):     {lbl['best_cut']:.3f}  ({lbl['best_cut']/exact_cut*100:.2f}%)")
    print(f"                   time: {lbl['time']:.1f}s  evals: {lbl['n_cost_evals']}")
    delta = valley['best_cut'] - lbl['best_cut']
    if delta > 0:
        print(f"  → Valley wins by +{delta:.3f}")
    elif delta < 0:
        print(f"  → LBL wins by +{-delta:.3f}")
    else:
        print(f"  → Tie (Δ = 0)")
    print("=" * 70)
    print(f"\n  Output files:")
    print(f"    graph_20node.png")
    print(f"    valley_detection_20node.png")
    print(f"    valley_refine_20node.png")
    print(f"    lbl_process_20node.png")
    print(f"    landscape_p1_20node.png")
    print(f"    landscape_p2_20node.png")
    print(f"    head_to_head_20node.png")
    print("\nDone.")


if __name__ == '__main__':
    main()
