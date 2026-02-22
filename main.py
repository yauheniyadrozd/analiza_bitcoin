"""KOMPLEKSOWA ANALIZA SIECI BITCOIN OTC TRUST NETWORK
Analiza obejmuje:
1. Podstawowe statystyki sieci
2. Wykrywanie hubów i klik
3. Analiza centralności
4. Sprawdzanie grafu Eulerowskiego
5. Przepływy maksymalne (Ford-Fulkerson)
6. Graf dwudzielny
7. Analiza sentymentu
8. Reprezentacje macierzowe
9. Wizualizacje
10. Dodatkowe metryki sieciowe"""
import gzip
import pandas as pd
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

INPUT_FILE = r'C:\pwr\3 sem\pycharm\bitcoin otc\bitcoin otc\soc-sign-bitcoinotc.csv.gz'
OUTPUT_DIR = r'C:\pwr\3 sem\pycharm\bitcoin otc\results_bitcoinotc'

def load_data(filepath):
    """Wczytaj dane z pliku .gz"""
    print("1: WCZYTYWANIE DANYCH")

    edges = []
    with gzip.open(filepath, 'rt') as f:
        for line in f:
            parts = line.strip().split(',')
            source, target, rating, timestamp = int(parts[0]), int(parts[1]), int(parts[2]), float(parts[3])
            edges.append((source, target, rating, timestamp))

    df = pd.DataFrame(edges, columns=['source', 'target', 'rating', 'timestamp'])
    print(f" Wczytano {len(df)} krawędzi")
    print(f" Liczba unikalnych wierzchołków: {len(set(df['source']) | set(df['target']))}")

    return df

# TWORZENIE GRAFU
def create_graph(df):
    """Utwórz graf NetworkX z danych"""
    print("2: TWORZENIE GRAFU")

    G = nx.DiGraph()
    for _, row in df.iterrows():
        G.add_edge(row['source'], row['target'],
                   weight=row['rating'],
                   timestamp=row['timestamp'])

    print(f"Graf utworzony: {G.number_of_nodes()} wierzchołków, {G.number_of_edges()} krawędzi")
    print(f"Gęstość grafu: {nx.density(G):.6f}")

    return G

# PODSTAWOWE STATYSTYKI
def basic_statistics(G):
    """Oblicz podstawowe statystyki sieci"""
    print("3: PODSTAWOWE STATYSTYKI SIECI")

    print(f"Czy graf jest silnie spójny: {nx.is_strongly_connected(G)}")
    print(f"Czy graf jest słabo spójny: {nx.is_weakly_connected(G)}")

    components = list(nx.weakly_connected_components(G))
    print(f"Liczba składowych słabo spójnych: {len(components)}")
    if len(components) > 1:
        largest_cc = max(components, key=len)
        print(f"Rozmiar największej składowej: {len(largest_cc)}")
        G_main = G.subgraph(largest_cc).copy()
    else:
        G_main = G

    in_degrees = dict(G.in_degree())
    out_degrees = dict(G.out_degree())

    print(f"\nŚredni stopień wejściowy: {np.mean(list(in_degrees.values())):.2f}")
    print(f"Średni stopień wyjściowy: {np.mean(list(out_degrees.values())):.2f}")

    return G_main, in_degrees, out_degrees

# HUBY I KLIKI
def analyze_hubs_and_cliques(G, in_degrees, out_degrees):
    """Wykryj huby i kliki"""
    print("4: ANALIZA HUBÓW I KLIK")

    # Huby
    total_degrees = {node: in_degrees[node] + out_degrees[node] for node in G.nodes()}
    top_hubs = sorted(total_degrees.items(), key=lambda x: x[1], reverse=True)[:10]

    print("\nTop 10 HUBÓW (największy stopień całkowity):")
    for i, (node, degree) in enumerate(top_hubs, 1):
        print(
            f"{i:2d}. Wierzchołek {int(node):4d}: stopień={degree:4d} (in={in_degrees[node]:3d}, out={out_degrees[node]:3d})")

    # Kliki
    G_undirected = G.to_undirected()

    if G_undirected.number_of_nodes() < 500:
        cliques = list(nx.find_cliques(G_undirected))
        max_clique = max(cliques, key=len)
        print(f"\nLiczba maksymalnych klik: {len(cliques)}")
        print(f"Rozmiar największej kliki: {len(max_clique)}")
    else:
        sample_nodes = list(G_undirected.nodes())[:500]
        G_sample = G_undirected.subgraph(sample_nodes)
        cliques = list(nx.find_cliques(G_sample))
        max_clique = max(cliques, key=len) if cliques else []
        print(f"\nAnaliza na próbce 500 wierzchołków:")
        print(f"Liczba maksymalnych klik w próbce: {len(cliques)}")
        print(f"Rozmiar największej kliki w próbce: {len(max_clique)}")

    # Współczynnik klasteryzacji
    clustering = nx.average_clustering(G_undirected)
    print(f"\nŚredni współczynnik klasteryzacji: {clustering:.6f}")

    return top_hubs, G_undirected

# CENTRALNOŚC
def analyze_centrality(G):
    """Oblicz miary centralności"""
    print("5: ANALIZA CENTRALNOŚCI")

    # Degree centrality
    degree_cent = nx.degree_centrality(G)
    top_degree = sorted(degree_cent.items(), key=lambda x: x[1], reverse=True)[:5]
    print("\nTop 5 wg Degree Centrality:")
    for node, cent in top_degree:
        print(f"  Wierzchołek {int(node)}: {cent:.4f}")

    # PageRank
    try:
        pagerank = nx.pagerank(G, max_iter=200, tol=1e-6)
        top_pagerank = sorted(pagerank.items(), key=lambda x: x[1], reverse=True)[:5]
        print("\nTop 5 wg PageRank:")
        for node, cent in top_pagerank:
            print(f"  Wierzchołek {int(node)}: {cent:.6f}")
    except:
        print("\nPageRank: nie udało się obliczyć (problem ze zbieżnością)")


# GRAF EULEROWSKI
def check_eulerian(G, G_undirected):
    """Sprawdź warunki grafu Eulerowskiego"""
    print("6: SPRAWDZANIE GRAFU EULEROWSKIEGO")

    # Dla grafu nieskierowanego
    print("\nDla grafu nieskierowanego:")
    if nx.is_eulerian(G_undirected):
        print(" Graf jest Eulerowski (istnieje cykl Eulera)")
        euler_circuit = list(nx.eulerian_circuit(G_undirected))
        print(f"  Długość cyklu: {len(euler_circuit)}")
    elif nx.is_semieulerian(G_undirected):
        print(" Graf jest półeulerowski (istnieje ścieżka Eulera)")
        euler_path = list(nx.eulerian_path(G_undirected))
        print(f"  Długość ścieżki: {len(euler_path)}")
        print("  Pierwsze 20 wierzchołków ścieżki:")
        vertices = [u for u, v in euler_path[:20]]
        print(f"  {' -> '.join(map(str, vertices))}")
    else:
        print(" Graf nie jest Eulerowski ani półeulerowski")
        odd_degree_nodes = [n for n in G_undirected.nodes() if G_undirected.degree(n) % 2 == 1]
        print(f"  Liczba wierzchołków o nieparzystym stopniu: {len(odd_degree_nodes)}")

    # Dla grafu skierowanego
    print("\nDla grafu skierowanego:")
    if nx.is_eulerian(G):
        print(" Graf jest Eulerowski (istnieje cykl Eulera)")
        euler_circuit = list(nx.eulerian_circuit(G))
        print(f"  Długość cyklu: {len(euler_circuit)}")
        print("  Pierwsze 20 krawędzi:")
        for i, (u, v) in enumerate(euler_circuit[:20], 1):
            print(f"    {i}. {u} -> {v}")
    elif nx.is_semieulerian(G):
        print(" Graf jest półeulerowski (istnieje ścieżka Eulera)")
        euler_path = list(nx.eulerian_path(G))
        print(f"  Długość ścieżki: {len(euler_path)}")
        print("  Pierwsze 20 krawędzi ścieżki:")
        for i, (u, v) in enumerate(euler_path[:20], 1):
            print(f"    {i}. {u} -> {v}")
    else:
        print("✗ Graf nie jest Eulerowski ani półeulerowski")
        balanced_nodes = [n for n in G.nodes() if G.in_degree(n) == G.out_degree(n)]
        print(f"  Wierzchołki zrównoważone (in=out): {len(balanced_nodes)}/{G.number_of_nodes()}")


# SEKCJA 7: PRZEPŁYWY MAKSYMALNE (FORD-FULKERSON)
def analyze_max_flow(G, top_hubs):
    """Oblicz przepływy maksymalne między hubami"""
    print("7: PRZEPŁYWY MAKSYMALNE (FORD-FULKERSON)")

    # Utwórz graf przepływowy (tylko pozytywne wagi)
    G_flow = nx.DiGraph()
    for u, v, data in G.edges(data=True):
        weight = data['weight']
        if weight > 0:
            G_flow.add_edge(u, v, capacity=weight)

    print(f"Graf przepływowy: {G_flow.number_of_nodes()} wierzchołków, {G_flow.number_of_edges()} krawędzi")
    print("(tylko pozytywne ratings jako przepustowość)\n")

    # Oblicz przepływy dla kilku par hubów
    test_pairs = [
        (int(top_hubs[0][0]), int(top_hubs[1][0])),
        (int(top_hubs[2][0]), int(top_hubs[4][0])),
        (int(top_hubs[4][0]), int(top_hubs[9][0])),
    ]

    for i, (source, target) in enumerate(test_pairs, 1):
        print(f"{i}. PRZEPŁYW: Hub #{source} → Hub #{target}")
        try:
            flow_value, flow_dict = nx.maximum_flow(G_flow, source, target)
            used_edges = sum(1 for u in flow_dict for v in flow_dict[u] if flow_dict[u][v] > 0)

            print(f"   Maksymalny przepływ: {flow_value}")
            print(f"   Wykorzystane krawędzie: {used_edges}")

            # Pokaż top 5 najsilniejszych przepływów
            edge_flows = [(u, v, flow_dict[u][v]) for u in flow_dict for v in flow_dict[u] if flow_dict[u][v] > 0]
            edge_flows.sort(key=lambda x: x[2], reverse=True)

            print("   Top 5 najsilniejszych przepływów:")
            for u, v, f in edge_flows[:5]:
                capacity = G_flow[u][v]['capacity']
                print(f"     {int(u)} → {int(v)}: przepływ={f:.1f}, przepustowość={capacity}")
            print()

        except nx.NetworkXError as e:
            print(f"    Nie można obliczyć: {e}\n")

    return G_flow

# GRAF DWUDZIELNY
def analyze_bipartite(G, df):
    """Analiza grafu dwudzielnego"""
    print("8: ANALIZA GRAFU DWUDZIELNEGO")

    G_undirected = G.to_undirected()
    is_bipartite = nx.is_bipartite(G_undirected)
    print(f"Czy oryginalny graf jest dwudzielny: {is_bipartite}")

    if not is_bipartite:
        print("\nTworzenie reprezentacji dwudzielnej: oceniający vs oceniani")
        B = nx.Graph()

        # Tylko pozytywne oceny
        positive_df = df[df['rating'] > 0]
        for _, row in positive_df.iterrows():
            rater = f"rater_{row['source']}"
            rated = f"rated_{row['target']}"
            B.add_node(rater, bipartite=0)
            B.add_node(rated, bipartite=1)
            B.add_edge(rater, rated, weight=row['rating'])

        print(f"✓ Graf dwudzielny utworzony: {B.number_of_nodes()} wierzchołków, {B.number_of_edges()} krawędzi")

        if nx.is_bipartite(B):
            raters = {n for n, d in B.nodes(data=True) if d['bipartite'] == 0}
            rated = {n for n, d in B.nodes(data=True) if d['bipartite'] == 1}
            print(f"  Zbiór oceniających: {len(raters)}")
            print(f"  Zbiór ocenianych: {len(rated)}")

        return B

    return None

# 9: ANALIZA SENTYMENTU
def analyze_sentiment(df):
    """Analiza sentymentu (ratingów)"""
    print("9: ANALIZA SENTYMENTU")

    print(f"Średni rating: {df['rating'].mean():.2f}")
    print(f"Mediana ratingu: {df['rating'].median():.0f}")
    print(f"Rating min: {df['rating'].min()}, max: {df['rating'].max()}")
    print(f"Odchylenie standardowe: {df['rating'].std():.2f}")

    positive = df[df['rating'] > 0]
    negative = df[df['rating'] < 0]
    neutral = df[df['rating'] == 0]

    print(f"\n Pozytywne oceny: {len(positive):5d} ({100 * len(positive) / len(df):5.1f}%)")
    print(f" Negatywne oceny: {len(negative):5d} ({100 * len(negative) / len(df):5.1f}%)")
    print(f" Neutralne oceny:  {len(neutral):5d} ({100 * len(neutral) / len(df):5.1f}%)")

    print(f"\nRozkład ratingów:")
    rating_counts = df['rating'].value_counts().sort_index()
    for rating, count in rating_counts.items():
        bar_length = int(count / 1000)
        bar = '-' * bar_length
        print(f"  {rating:+3d}: {count:5d} {bar}")

# 10: REPREZENTACJE MACIERZOWE
def create_matrices(G):
    """Utwórz macierze incydencji i sąsiedztwa"""
    print("10: REPREZENTACJE MACIERZOWE")

    # Podgraf dla macierzy (dla wydajności)
    subgraph_nodes = list(G.nodes())[:100]
    G_sub = G.subgraph(subgraph_nodes).copy()

    print(f"Podgraf: {G_sub.number_of_nodes()} wierzchołków, {G_sub.number_of_edges()} krawędzi")

    # Macierz sąsiedztwa
    adj_matrix = nx.to_numpy_array(G_sub)
    print(f"\n Macierz sąsiedztwa: wymiar {adj_matrix.shape}")
    print("  Przykład (pierwsze 5x5):")
    print(adj_matrix[:5, :5])

    # Macierz incydencji
    incidence_matrix = nx.incidence_matrix(G_sub, oriented=True)
    print(f"\n✓ Macierz incydencji: wymiar {incidence_matrix.shape}")

    return G_sub, adj_matrix

# 11: DODATKOWE ANALIZY
def additional_analyses(G):
    """Dodatkowe analizy sieciowe"""
    print("11: DODATKOWE ANALIZY")

    # Assortatywność
    assortativity = nx.degree_assortativity_coefficient(G)
    print(f"\n1. Assortatywność stopnia: {assortativity:.4f}")
    if assortativity > 0:
        print("   Assortative: huby łączą się z hubami")
    else:
        print("   Disassortative: huby łączą się z małymi węzłami")

    # Reciprocity
    reciprocity = nx.reciprocity(G)
    print(f"\n2. Reciprocity (wzajemność): {reciprocity:.4f}")
    print(f"   → {reciprocity * 100:.1f}% krawędzi ma krawędź zwrotną")

    # Średnica (próbka)
    sample_nodes = list(G.nodes())[:300]
    G_sample = G.subgraph(sample_nodes).to_undirected()
    try:
        diameter = nx.diameter(G_sample)
        avg_path = nx.average_shortest_path_length(G_sample)
        print(f"\n3. Średnica sieci (próbka 300): {diameter}")
        print(f"   Średnia długość ścieżki: {avg_path:.2f}")
    except:
        print(f"\n3. Średnica: nie można obliczyć (graf niespójny)")

    # K-core
    G_undirected = G.to_undirected()
    core_numbers = nx.core_number(G_undirected)
    max_k = max(core_numbers.values())
    k_core = nx.k_core(G_undirected)
    print(f"\n4. K-core decomposition:")
    print(f"   Maksymalne k: {max_k}")
    print(f"   Rozmiar głównego k-core: {k_core.number_of_nodes()} węzłów")

    # Mosty
    bridges = list(nx.bridges(G_undirected))
    print(f"\n5. Mosty (krawędzie krytyczne): {len(bridges)}")
    if len(bridges) > 0 and len(bridges) <= 10:
        print("   Przykłady:")
        for u, v in bridges[:10]:
            print(f"   - {int(u)} - {int(v)}")

    # Analiza in/out
    in_deg = dict(G.in_degree())
    out_deg = dict(G.out_degree())
    only_in = sum(1 for n in G.nodes() if in_deg[n] > 0 and out_deg[n] == 0)
    only_out = sum(1 for n in G.nodes() if out_deg[n] > 0 and in_deg[n] == 0)
    both = sum(1 for n in G.nodes() if in_deg[n] > 0 and out_deg[n] > 0)

    print(f"\n6. Analiza kierunkowości:")
    print(f"   Tylko otrzymujące: {only_in} ({100 * only_in / G.number_of_nodes():.1f}%)")
    print(f"   Tylko wystawiające: {only_out} ({100 * only_out / G.number_of_nodes():.1f}%)")
    print(f"   Dwukierunkowe: {both} ({100 * both / G.number_of_nodes():.1f}%)")


# GRAF DWUDZIELNY
def analyze_bipartite(G, df):
    """Analiza grafu dwudzielnego"""
    print("8: ANALIZA GRAFU DWUDZIELNEGO")

    G_undirected = G.to_undirected()
    is_bipartite = nx.is_bipartite(G_undirected)
    print(f"Czy oryginalny graf jest dwudzielny: {is_bipartite}")

    if not is_bipartite:
        print("\nTworzenie reprezentacji dwudzielnej: oceniający vs oceniani")
        B = nx.Graph()

        # Tylko pozytywne oceny
        positive_df = df[df['rating'] > 0]
        for _, row in positive_df.iterrows():
            rater = f"rater_{row['source']}"
            rated = f"rated_{row['target']}"
            B.add_node(rater, bipartite=0)
            B.add_node(rated, bipartite=1)
            B.add_edge(rater, rated, weight=row['rating'])

        print(f"✓ Graf dwudzielny utworzony: {B.number_of_nodes()} wierzchołków, {B.number_of_edges()} krawędzi")

        if nx.is_bipartite(B):
            raters = {n for n, d in B.nodes(data=True) if d['bipartite'] == 0}
            rated = {n for n, d in B.nodes(data=True) if d['bipartite'] == 1}
            print(f"  Zbiór oceniających: {len(raters)}")
            print(f"  Zbiór ocenianych: {len(rated)}")

        return B

    return None


# WIZUALIZACJE
def create_visualizations(G, df, top_hubs, adj_matrix, B, output_dir):
    """Utwórz wizualizacje"""
    print("12: GENEROWANIE WIZUALIZACJI")

    # === WIZUALIZACJA 1: Rozkłady ===
    print("\n1. Generowanie wykresów rozkładów...")
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Bitcoin OTC Network - Podstawowe Rozkłady', fontsize=16, fontweight='bold')

    # Stopnie wejściowe
    in_degrees = [d for n, d in G.in_degree()]
    axes[0, 0].hist(in_degrees, bins=50, color='steelblue', alpha=0.7, edgecolor='black')
    axes[0, 0].set_xlabel('Stopień wejściowy')
    axes[0, 0].set_ylabel('Liczba wierzchołków')
    axes[0, 0].set_yscale('log')
    axes[0, 0].set_title('Rozkład stopni wejściowych')
    axes[0, 0].grid(True, alpha=0.3)

    # Stopnie wyjściowe
    out_degrees = [d for n, d in G.out_degree()]
    axes[0, 1].hist(out_degrees, bins=50, color='coral', alpha=0.7, edgecolor='black')
    axes[0, 1].set_xlabel('Stopień wyjściowy')
    axes[0, 1].set_ylabel('Liczba wierzchołków')
    axes[0, 1].set_yscale('log')
    axes[0, 1].set_title('Rozkład stopni wyjściowych')
    axes[0, 1].grid(True, alpha=0.3)

    # Ratingi
    axes[1, 0].hist(df['rating'], bins=21, color='green', alpha=0.7, edgecolor='black')
    axes[1, 0].set_xlabel('Rating')
    axes[1, 0].set_ylabel('Liczba krawędzi')
    axes[1, 0].set_title('Rozkład ratingów (sentyment)')
    axes[1, 0].axvline(x=0, color='red', linestyle='--', linewidth=2)
    axes[1, 0].grid(True, alpha=0.3)

    # Top 20 hubów
    top_20 = top_hubs[:20]
    nodes = [str(int(n)) for n, _ in top_20]
    degrees = [d for _, d in top_20]
    axes[1, 1].barh(nodes[::-1], degrees[::-1], color='purple', alpha=0.7)
    axes[1, 1].set_xlabel('Stopień całkowity')
    axes[1, 1].set_ylabel('Wierzchołek')
    axes[1, 1].set_title('Top 20 Hubów')
    axes[1, 1].grid(True, alpha=0.3, axis='x')

    plt.tight_layout()
    plt.savefig(f'{output_dir}viz1_distributions.png', dpi=200, bbox_inches='tight')
    print(f"    Zapisano: viz1_distributions.png")
    plt.close()

    # === WIZUALIZACJA 2: Sieć hubów ===
    print("2. Generowanie wizualizacji sieci hubów...")
    top_50 = [n for n, _ in top_hubs[:50]]
    neighbors = set(top_50)
    for node in top_50:
        neighbors.update(list(G.predecessors(node))[:5])
        neighbors.update(list(G.successors(node))[:5])
    neighbors = list(neighbors)[:150]
    G_viz = G.subgraph(neighbors).copy()

    plt.figure(figsize=(18, 18))
    pos = nx.spring_layout(G_viz, k=2, iterations=30, seed=42)
    node_sizes = [20 * (G_viz.degree(n) + 1) for n in G_viz.nodes()]

    weights = [G_viz[u][v]['weight'] for u, v in G_viz.edges()]
    edge_colors = ['green' if w > 0 else 'red' for w in weights]

    nx.draw_networkx_nodes(G_viz, pos, node_size=node_sizes,
                           node_color='lightblue', alpha=0.6, edgecolors='black')
    nx.draw_networkx_edges(G_viz, pos, edge_color=edge_colors,
                           alpha=0.2, arrows=True, arrowsize=8)

    # Oznacz top 10
    top_10_in_viz = [n for n in top_50 if n in G_viz.nodes()][:10]
    nx.draw_networkx_labels(G_viz, {n: pos[n] for n in top_10_in_viz},
                            {n: str(int(n)) for n in top_10_in_viz}, font_size=8)

    plt.title('Bitcoin OTC - Sieć Top Hubów\n(Zielone=pozytywne, Czerwone=negatywne)',
              fontsize=14, fontweight='bold')
    plt.axis('off')
    plt.savefig(f'{output_dir}viz2_hub_network.png', dpi=200, bbox_inches='tight')
    print(f"    Zapisano: viz2_hub_network.png")
    plt.close()

    # === WIZUALIZACJA 3: Macierz sąsiedztwa ===
    print("3. Generowanie heatmapy macierzy sąsiedztwa...")
    plt.figure(figsize=(12, 10))
    sns.heatmap(adj_matrix, cmap='RdYlGn', center=0,
                cbar_kws={'label': 'Rating'},
                xticklabels=False, yticklabels=False)
    plt.title('Macierz Sąsiedztwa (100 węzłów)\nKolor: Zielony=pozytywny, Czerwony=negatywny',
              fontsize=14, fontweight='bold')
    plt.xlabel('Wierzchołek docelowy')
    plt.ylabel('Wierzchołek źródłowy')
    plt.tight_layout()
    plt.savefig(f'{output_dir}viz3_adjacency_matrix.png', dpi=200, bbox_inches='tight')
    print(f"    Zapisano: viz3_adjacency_matrix.png")
    plt.close()

    # === WIZUALIZACJA 4: Graf dwudzielny ===
    if B is not None:
        print("4. Generowanie wizualizacji grafu dwudzielnego...")

        # Wybierz podzbiór grafu do wizualizacji (dla czytelności)
        # Top węzły według stopnia
        degrees = dict(B.degree())
        top_nodes = sorted(degrees.items(), key=lambda x: x[1], reverse=True)[:100]
        top_node_names = [n for n, _ in top_nodes]
        B_viz = B.subgraph(top_node_names).copy()

        # Rozdziel węzły na dwa zbiory
        raters = [n for n, d in B_viz.nodes(data=True) if d.get('bipartite') == 0]
        rated = [n for n, d in B_viz.nodes(data=True) if d.get('bipartite') == 1]

        # Układ dwudzielny
        pos = {}

        # Oceniający po lewej stronie
        for i, node in enumerate(raters):
            pos[node] = (0, i * (len(rated) / len(raters)) if len(raters) > 0 else i)

        # Oceniani po prawej stronie
        for i, node in enumerate(rated):
            pos[node] = (1, i)

        # Rysuj wykres
        plt.figure(figsize=(16, 20))

        # Rozmiary węzłów proporcjonalne do stopnia
        node_sizes_raters = [50 * (B_viz.degree(n) + 1) for n in raters]
        node_sizes_rated = [50 * (B_viz.degree(n) + 1) for n in rated]

        # Rysuj węzły - oceniający (niebieski)
        nx.draw_networkx_nodes(B_viz, pos, nodelist=raters,
                               node_size=node_sizes_raters,
                               node_color='lightblue',
                               alpha=0.7,
                               edgecolors='darkblue',
                               linewidths=1.5,
                               label='Oceniający')

        # Rysuj węzły - oceniani (pomarańczowy)
        nx.draw_networkx_nodes(B_viz, pos, nodelist=rated,
                               node_size=node_sizes_rated,
                               node_color='lightcoral',
                               alpha=0.7,
                               edgecolors='darkred',
                               linewidths=1.5,
                               label='Oceniani')

        # Rysuj krawędzie z gradientem koloru według wagi
        edges = B_viz.edges(data=True)
        weights = [e[2].get('weight', 1) for e in edges]

        nx.draw_networkx_edges(B_viz, pos,
                               edge_color=weights,
                               edge_cmap=plt.cm.YlGn,
                               alpha=0.3,
                               width=0.5)

        # Dodaj etykiety dla top 10 węzłów w każdym zbiorze
        top_raters = sorted(raters, key=lambda n: B_viz.degree(n), reverse=True)[:10]
        top_rated = sorted(rated, key=lambda n: B_viz.degree(n), reverse=True)[:10]

        labels = {}
        for n in top_raters:
            labels[n] = n.replace('rater_', 'R')
        for n in top_rated:
            labels[n] = n.replace('rated_', 'D')

        nx.draw_networkx_labels(B_viz, pos, labels, font_size=6, font_weight='bold')

        plt.title(f'Graf Dwudzielny Bitcoin OTC\n'
                  f'Oceniający ({len(raters)}) ← → Oceniani ({len(rated)})\n'
                  f'(Top 100 węzłów według stopnia)',
                  fontsize=14, fontweight='bold', pad=20)

        plt.legend(loc='upper right', fontsize=10)
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(f'{output_dir}viz4_bipartite_graph.png', dpi=200, bbox_inches='tight')
        print(f"    Zapisano: viz4_bipartite_graph.png")
        plt.close()

        # === WIZUALIZACJA 4B: Statystyki grafu dwudzielnego ===
        print("5. Generowanie statystyk grafu dwudzielnego...")

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Graf Dwudzielny - Statystyki', fontsize=16, fontweight='bold')

        # Rozkład stopni - oceniający
        rater_degrees = [B.degree(n) for n in raters if n in B.nodes()]
        axes[0, 0].hist(rater_degrees, bins=30, color='steelblue', alpha=0.7, edgecolor='black')
        axes[0, 0].set_xlabel('Stopień węzła')
        axes[0, 0].set_ylabel('Liczba węzłów')
        axes[0, 0].set_title('Rozkład stopni - Oceniający')
        axes[0, 0].set_yscale('log')
        axes[0, 0].grid(True, alpha=0.3)

        # Rozkład stopni - oceniani
        rated_degrees = [B.degree(n) for n in rated if n in B.nodes()]
        axes[0, 1].hist(rated_degrees, bins=30, color='coral', alpha=0.7, edgecolor='black')
        axes[0, 1].set_xlabel('Stopień węzła')
        axes[0, 1].set_ylabel('Liczba węzłów')
        axes[0, 1].set_title('Rozkład stopni - Oceniani')
        axes[0, 1].set_yscale('log')
        axes[0, 1].grid(True, alpha=0.3)

        # Porównanie średnich stopni
        avg_rater = np.mean(rater_degrees) if rater_degrees else 0
        avg_rated = np.mean(rated_degrees) if rated_degrees else 0
        axes[1, 0].bar(['Oceniający', 'Oceniani'], [avg_rater, avg_rated],
                       color=['steelblue', 'coral'], alpha=0.7, edgecolor='black')
        axes[1, 0].set_ylabel('Średni stopień')
        axes[1, 0].set_title('Porównanie średnich stopni')
        axes[1, 0].grid(True, alpha=0.3, axis='y')

        # Top 15 najbardziej aktywnych węzłów
        all_degrees = [(n, B.degree(n)) for n in B.nodes()]
        top_15 = sorted(all_degrees, key=lambda x: x[1], reverse=True)[:15]

        nodes_15 = [n.replace('rater_', 'R').replace('rated_', 'D')[:10] for n, _ in top_15]
        degrees_15 = [d for _, d in top_15]
        colors_15 = ['steelblue' if n.startswith('R') else 'coral' for n, _ in top_15]

        axes[1, 1].barh(nodes_15[::-1], degrees_15[::-1], color=colors_15[::-1], alpha=0.7)
        axes[1, 1].set_xlabel('Stopień')
        axes[1, 1].set_title('Top 15 najbardziej aktywnych węzłów')
        axes[1, 1].grid(True, alpha=0.3, axis='x')

        plt.tight_layout()
        plt.savefig(f'{output_dir}viz5_bipartite_stats.png', dpi=200, bbox_inches='tight')
        print(f"    Zapisano: viz5_bipartite_stats.png")
        plt.close()

    print("\n✓ Wszystkie wizualizacje wygenerowane!")


# FUNKCJA GŁÓWNA
def main():
    """Główna funkcja uruchamiająca wszystkie analizy"""
    # 1. Wczytaj dane
    df = load_data(INPUT_FILE)

    # 2. Utwórz graf
    G = create_graph(df)

    # 3. Podstawowe statystyki
    G_main, in_degrees, out_degrees = basic_statistics(G)

    # 4. Huby i kliki
    top_hubs, G_undirected = analyze_hubs_and_cliques(G, in_degrees, out_degrees)

    # 5. Centralności
    analyze_centrality(G)

    # 6. Graf Eulerowski
    check_eulerian(G, G_undirected)

    # 7. Przepływy maksymalne
    G_flow = analyze_max_flow(G, top_hubs)

    # 8. Graf dwudzielny
    B = analyze_bipartite(G, df)

    # 9. Analiza sentymentu
    analyze_sentiment(df)

    # 10. Macierze
    G_sub, adj_matrix = create_matrices(G)

    # 11. Dodatkowe analizy
    additional_analyses(G)

    # 12. Wizualizacje (teraz z grafem dwudzielnym)
    create_visualizations(G, df, top_hubs, adj_matrix, B, OUTPUT_DIR)

    # Podsumowanie
    print("\n" + "=" * 80)
    print("PODSUMOWANIE ANALIZY")
    print("=" * 80)
    print(f"""
     Wierzchołki: {G.number_of_nodes()}
     Krawędzie: {G.number_of_edges()}
     Największy hub: #{int(top_hubs[0][0])} (stopień: {top_hubs[0][1]})
     Pozytywne oceny: {len(df[df['rating'] > 0])} ({100 * len(df[df['rating'] > 0]) / len(df):.1f}%)
     Graf Eulerowski: NIE
     Graf dwudzielny: TAK (reprezentacja utworzona)
     Przepływy: Obliczone dla top hubów
     Wizualizacje: 5 plików PNG wygenerowanych
    """)

    print("✓ ANALIZA ZAKOŃCZONA!")

    print("\n")

    print("Pliki wyjściowe:")
    print("  - viz1_distributions.png - Rozkłady i statystyki")
    print("  - viz2_hub_network.png - Sieć hubów")
    print("  - viz3_adjacency_matrix.png - Macierz sąsiedztwa")
    print("  - viz4_bipartite_graph.png - Wizualizacja grafu dwudzielnego")
    print("  - viz5_bipartite_stats.png - Statystyki grafu dwudzielnego")
    print("\nAby obliczyć przepływ między konkretnymi węzłami, użyj:")
    print("  python max_flow_interactive.py <źródło> <cel>")


if __name__ == "__main__":
    main()