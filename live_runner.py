import sys
import os
import time
import subprocess

# --- AUTO-INSTALACJA ---
def install(package):
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
    except: pass

try:
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    import pandas as pd
    import numpy as np
    from sklearn.decomposition import PCA
    from matplotlib.ticker import StrMethodFormatter, PercentFormatter
except ImportError:
    print("Instalowanie bibliotek...")
    install("matplotlib")
    install("pandas")
    install("scikit-learn")
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    import pandas as pd
    import numpy as np
    from sklearn.decomposition import PCA
    from matplotlib.ticker import StrMethodFormatter, PercentFormatter

# --- KONFIGURACJA ---
FILE_VIEW = "live_view.txt"
FILE_HISTORY = "history.csv"
FILE_STATS = "mutation_stats.txt"
FILE_TRAJ = "trajectory.csv"
FILE_META = "active_instance.txt"

# 10 kolumn z C++ (zgodne z Twoim kodem)
COLS = ["Gen", "Fit", "IsCat", "CV", "MutRate", "VNDProb", "RuinProb", "Evals", "Hits", "UnqSols"]

plt.style.use('dark_background')

def run_dashboard():
    fig = plt.figure(figsize=(18, 10))
    fig.canvas.manager.set_window_title('LcVRP Ultimate Dashboard v3.2 (Params View)')

    # Układ
    gs = gridspec.GridSpec(2, 3, width_ratios=[2.2, 1.8, 1.2], height_ratios=[1, 1])

    ax_map = fig.add_subplot(gs[:, 0])      # Mapa
    ax_fit = fig.add_subplot(gs[0, 1])      # Fitness
    ax_probs = fig.add_subplot(gs[1, 1])    # ZMIANA: Parametry (zamiast CV)
    ax_stat = fig.add_subplot(gs[0, 2])     # Statsy Operatorów
    ax_pca  = fig.add_subplot(gs[1, 2])     # PCA

    pca_engine = PCA(n_components=2)
    
    print("[Python] Dashboard gotowy.")

    while True:
        try:
            # 1. HISTORIA
            if os.path.exists(FILE_HISTORY):
                try:
                    df = pd.read_csv(FILE_HISTORY, names=COLS, header=None, on_bad_lines='skip')
                    df = df.apply(pd.to_numeric, errors='coerce').dropna()
                    
                    if len(df) > 5:
                        last = df.iloc[-1]
                        
                        # --- NAGŁÓWEK ---
                        inst_name = "Unknown"
                        if os.path.exists(FILE_META):
                            try:
                                with open(FILE_META) as f: inst_name = f.read().strip()
                            except: pass
                        
                        hits = int(last['Hits'])
                        evals = int(last['Evals'])
                        hit_ratio = (hits / evals * 100.0) if evals > 0 else 0.0
                        
                        header = (f"INSTANCE: {inst_name}  |  GEN: {int(last['Gen'])}  |  FIT: {last['Fit']:,.0f}\n"
                                  f"Unique Sols: {int(last['UnqSols'])}  |  Cache Hits: {hit_ratio:.1f}%  |  CV: {last['CV']:.2e}")
                        
                        fig.suptitle(header, fontsize=14, color='yellow', fontweight='bold')

                        # --- WYKRES 1: FITNESS ---
                        ax_fit.clear()
                        ax_fit.plot(df["Gen"], df["Fit"], color='#00ff00', lw=1.5)
                        ax_fit.set_ylabel("Cost")
                        ax_fit.set_title("Fitness Trend", fontsize=10, color='gray')
                        ax_fit.grid(True, alpha=0.2)
                        ax_fit.yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))

                        # --- WYKRES 2: PARAMETRY ADAPTACYJNE (ZAMIAST CV) ---
                        ax_probs.clear()
                        
                        # Rysowanie linii parametrów
                        ax_probs.plot(df["Gen"], df["VNDProb"], color='cyan', lw=1.5, alpha=0.9, label="VND Prob")
                        ax_probs.plot(df["Gen"], df["MutRate"], color='orange', lw=1.5, alpha=0.8, label="Mut Rate")
                        ax_probs.plot(df["Gen"], df["RuinProb"], color='magenta', lw=1.2, alpha=0.6, ls=':', label="Ruin Prob")

                        # Rysowanie pionowych linii katastrof (jako kontekst)
                        cats = df[df["IsCat"] == 1]
                        if not cats.empty:
                            for g in cats["Gen"]:
                                ax_probs.axvline(x=g, c='red', alpha=0.4, ls='--', lw=1)
                            # Dodajemy jeden wpis do legendy dla katastrof
                            ax_probs.plot([], [], color='red', ls='--', label='Reset')

                        ax_probs.set_title("Adaptive Parameters & Resets", fontsize=10, color='white')
                        ax_probs.legend(loc='upper left', fontsize=8, framealpha=0.8)
                        ax_probs.grid(True, alpha=0.2)
                        
                        # Formatowanie osi Y jako procenty (0-100%)
                        ax_probs.set_ylim(0, 1.05)
                        ax_probs.yaxis.set_major_formatter(PercentFormatter(1.0))

                except Exception as e: pass

            # 2. PCA
            if os.path.exists(FILE_TRAJ):
                try:
                    traj = pd.read_csv(FILE_TRAJ, header=None, on_bad_lines='skip')
                    if len(traj) > 2000: traj = traj.sample(2000).sort_index()
                    traj = traj.dropna()

                    if len(traj) > 10:
                        fitness = traj.iloc[:, 1].values
                        genos = traj.iloc[:, 2:].values
                        coords = pca_engine.fit_transform(genos)
                        
                        ax_pca.clear()
                        
                        f_min, f_max = np.min(fitness), np.max(fitness)
                        if f_max > f_min:
                            c_val = (f_max - fitness) / (f_max - f_min) 
                        else: c_val = np.ones_like(fitness)

                        ax_pca.scatter(coords[:,0], coords[:,1], c=c_val, cmap='plasma', s=20, alpha=0.7, edgecolors='none')
                        ax_pca.scatter(coords[-1,0], coords[-1,1], c='cyan', marker='*', s=200, edgecolors='white', zorder=10)
                        
                        ax_pca.set_title("Search Space (PCA)", fontsize=9)
                        ax_pca.axis('off')
                except Exception: pass

            # 3. STATYSTYKI OPERATORÓW
            if os.path.exists(FILE_STATS):
                try:
                    sdf = pd.read_csv(FILE_STATS, on_bad_lines='skip')
                    if not sdf.empty:
                        sdf = sdf.sort_values('Rate')
                        ax_stat.clear()
                        colors = ['#ff4444' if r < 2.0 else '#00cc00' for r in sdf['Rate']]
                        bars = ax_stat.barh(sdf['Name'], sdf['Rate'], color=colors)
                        ax_stat.bar_label(bars, fmt='%.1f%%', padding=3, color='white', fontsize=9)
                        ax_stat.set_title("Op Success Rate", fontsize=10)
                        ax_stat.grid(axis='x', alpha=0.2)
                except: pass

            # 4. MAPA (Live View)
            if os.path.exists(FILE_VIEW):
                lines = []
                try: 
                    with open(FILE_VIEW, 'r') as f: 
                        lines = f.readlines()
                except: pass

                if lines:
                    ax_map.clear() # Czyścimy całość mapy dla bezpieczeństwa
                    
                    real_coords_exist = False
                    nodes = {} 
                    depot = None
                    
                    for line in lines:
                        if line.startswith('N'):
                            parts = line.split()
                            nid = int(parts[1])
                            x, y = float(parts[2]), float(parts[3])
                            is_depot = (float(parts[4]) == 0)
                            
                            if is_depot: depot = (x,y)
                            else: nodes[nid] = (x,y)
                            
                            # Jeśli współrzędne to nie same zera, to znaczy że mamy geometrię
                            if abs(x) > 0.001 or abs(y) > 0.001:
                                real_coords_exist = True

                    # Rysowanie
                    if real_coords_exist:
                        # Trasy
                        for line in lines:
                            if line.startswith('R'):
                                parts = line.split()
                                c = [float(x) for x in parts[1:]]
                                if len(c) >= 4:
                                    ax_map.plot(c[0::2], c[1::2], '-', color='white', lw=0.8, alpha=0.5)
                        
                        # Węzły
                        if nodes:
                            xs, ys = zip(*nodes.values())
                            ax_map.scatter(xs, ys, c='dodgerblue', s=10, alpha=0.7)
                        if depot:
                            ax_map.scatter([depot[0]], [depot[1]], c='red', marker='s', s=80, zorder=10)
                    else:
                        ax_map.text(0.5, 0.5, "NO GEOMETRY\n(Matrix Mode)", 
                                    ha='center', va='center', color='white', fontsize=12)

                    ax_map.axis('equal')
                    ax_map.set_xticks([]); ax_map.set_yticks([])
                    ax_map.grid(False)

            plt.pause(1.0)

        except Exception as e:
            print(f"Error: {e}")
            time.sleep(1)

if __name__ == "__main__":
    run_dashboard()