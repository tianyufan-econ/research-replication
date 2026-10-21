"""Label placement for Figure 17 (used by 10_accounting.py).
Copy of the full package's code/03_extensions_plot/label_layout_big.py (only these two lines differ).

Deterministic point-label placement: every label goes next to its own dot, on the side where the previous
(adjustText) version had it when that side is free; labels never overlap each other, a labelled dot or the
legend, and every other labelled dot is at least 3 pt farther from a label than its own dot. Faded (unlabelled)
dots are avoided where possible. The previous version's connector lines are kept for the same labels, drawn the
way adjustText draws them. No randomness and no time limit, so every run gives the same figure.
"""
import numpy as np


def _ray_start(w, h, u, rho):
    """smallest t >= 0 with dist(point, box centred at point + t*u) >= rho (box w x h)."""
    lo, hi = 0.0, rho + w + h + 1.0
    for _ in range(60):
        t = 0.5 * (lo + hi)
        dx = max(t * abs(u[0]) - w / 2, 0.0)
        dy = max(t * abs(u[1]) - h / 2, 0.0)
        if np.hypot(dx, dy) >= rho:
            hi = t
        else:
            lo = t
    return hi


def _dist_box_points(c, w, h, P):
    dx = np.maximum(np.abs(P[:, 0] - c[0]) - w / 2, 0.0)
    dy = np.maximum(np.abs(P[:, 1] - c[1]) - h / 2, 0.0)
    return np.hypot(dx, dy)


def candidates(targets, sizes, own, dots, dot_r, hard, legend_box, axes_box, prefer, max_ring,
               gap, margin, ring, ndir=16, ring_cost=1.0, faded_cost=1.0, assoc=0.0):
    """per label: list of (cost, ring, direction index, centre x, centre y, faded cover), sorted by cost."""
    out = []
    ax0, ay0, ax1, ay1 = axes_box
    for i, (T, (w, h)) in enumerate(zip(targets, sizes)):
        others = np.ones(len(dots), bool)
        others[own[i]] = False
        H, S = others & hard, others & ~hard
        cands = []
        for j in range(ndir):
            th = 2 * np.pi * j / ndir
            u = (np.cos(th), np.sin(th))
            t0 = _ray_start(w, h, u, dot_r[own[i]] + gap)
            for k in range(max_ring[i] + 1):
                t = t0 + k * ring
                c = (T[0] + t * u[0], T[1] + t * u[1])
                if c[0] - w / 2 < ax0 + margin or c[0] + w / 2 > ax1 - margin \
                        or c[1] - h / 2 < ay0 + margin or c[1] + h / 2 > ay1 - margin:
                    continue
                if legend_box is not None:
                    lx0, ly0, lx1, ly1 = legend_box
                    if c[0] + w / 2 + margin > lx0 and c[0] - w / 2 - margin < lx1 \
                            and c[1] + h / 2 + margin > ly0 and c[1] - h / 2 - margin < ly1:
                        continue
                dh = _dist_box_points(c, w, h, dots[H]) - dot_r[H]        # clearance to other labelled dots
                d_own = _dist_box_points(c, w, h, dots[own[i]][None])[0] - dot_r[own[i]]
                if np.any(dh < margin):
                    continue
                if len(dh) and dh.min() < d_own + assoc:   # another labelled dot (nearly) as close: ambiguous
                    continue
                depth = dot_r[S] - _dist_box_points(c, w, h, dots[S])
                cover = float(np.minimum(depth[depth > 0] / dot_r[S][depth > 0], 1.0).sum())
                ang = 0.0 if np.isnan(prefer[i]) else abs((th - prefer[i] + np.pi) % (2 * np.pi) - np.pi) / np.pi
                cands.append((k * ring_cost + ang + faded_cost * cover + 1e-6 * j, k, j, c[0], c[1], cover))
        cands.sort()
        out.append(cands)
    return out


def _boxes(cands, sizes, margin):
    return [np.array([[c[3] - sizes[i][0] / 2 - margin / 2, c[4] - sizes[i][1] / 2 - margin / 2,
                       c[3] + sizes[i][0] / 2 + margin / 2, c[4] + sizes[i][1] / 2 + margin / 2]
                      for c in cands[i]]).reshape(-1, 4) for i in range(len(cands))]


def _hit(b, P):
    return (b[0] < P[:, 2]) & (b[2] > P[:, 0]) & (b[1] < P[:, 3]) & (b[3] > P[:, 1])


def solve(cands, sizes, margin, max_nodes=2_000_000):
    """depth-first search (most constrained label first) + single-label improvement passes.
    Returns (chosen candidate index per label, search nodes, improvement moves)."""
    n = len(cands)
    order = sorted(range(n), key=lambda i: (len(cands[i]), i))
    boxes = _boxes(cands, sizes, margin)
    choice = [None] * n
    placed = []
    nodes = [0]

    def rec(pos):
        if pos == n:
            return True
        i = order[pos]
        P = np.array(placed) if placed else np.zeros((0, 4))
        for a in range(len(cands[i])):
            nodes[0] += 1
            if nodes[0] > max_nodes:
                raise RuntimeError('label search exceeded its node budget')
            if not np.any(_hit(boxes[i][a], P)):
                choice[i] = a
                placed.append(boxes[i][a])
                if rec(pos + 1):
                    return True
                placed.pop()
                choice[i] = None
        return False

    if not rec(0):
        raise RuntimeError('no overlap-free label placement with these rings')
    moves = 0
    changed = True
    while changed:
        changed = False
        for i in range(n):
            P = np.array([boxes[k][choice[k]] for k in range(n) if k != i])
            for a in range(choice[i]):                 # candidates cheaper than the current one
                if not np.any(_hit(boxes[i][a], P)):
                    choice[i] = a
                    moves += 1
                    changed = True
                    break
    return choice, nodes[0], moves


from matplotlib.patches import FancyArrowPatch

# Previous (adjustText) version, per figure: label centre in data coordinates and whether it had a connector.
PREVIOUS = {
    'geo_growth_period_scatter': {
        'AFG': ((-16.598552430837213, 14.684822843867508), False),
        'BLR': ((-7.683956114029783, -11.939705268520381), True),
        'BRA': ((-1.031728830322443, -1.7804346735162326), True),
        'CHL': ((-17.952756434485476, 24.3758997073039), False),
        'CHN': ((0.6913163249878451, -17.994600646288312), False),
        'CUB': ((-7.599931572791238, -0.5732333219000729), False),
        'DEU': ((5.654781121794869, -15.066634428219022), False),
        'ERI': ((-9.334303354848181, -15.651814306456359), True),
        'EST': ((-14.469857470962513, 17.72120181239788), True),
        'FRA': ((1.867830116675961, -2.6515772268891595), False),
        'GBR': ((4.315104427875781, -8.119834990258585), True),
        'GEO': ((-10.892862816388803, 16.19402727071052), False),
        'HUN': ((7.920902657076816, -15.976250141932542), False),
        'IND': ((3.287819562850551, -0.3634123994764984), True),
        'IRN': ((-9.655556806413173, -10.533940206380006), True),
        'JPN': ((1.0674192202532282, 0.5686286172027586), True),
        'KOR': ((-1.6543062814326817, 2.6884191523346033), False),
        'LBY': ((-23.47786963144064, 17.02535004499611), False),
        'LTU': ((-19.493213754227106, 22.37809542105535), False),
        'LVA': ((-16.5212559404686, 20.87232441515298), False),
        'MMR': ((-3.6484916586429, -13.092982445944259), True),
        'NIC': ((-1.0303553282533429, -15.011049800448415), False),
        'PRK': ((-17.548050020976618, -13.151969355868372), False),
        'QAT': ((-7.948641606872258, 13.321421510968953), False),
        'RUS': ((-9.06296376865302, -8.323632010600363), False),
        'SAU': ((3.389407954159317, -5.652579044514347), True),
        'SGP': ((5.770085127945286, 2.8757568605774395), False),
        'SYR': ((-6.366108744344597, -15.160965545250217), True),
        'TUR': ((-1.327634472067885, -5.884759060153957), False),
        'UKR': ((-9.651532512899337, 8.89205948893489), False),
        'USA': ((-4.417777925740982, -10.840864842965125), False),
        'VEN': ((-2.5131377935739394, -19.58279028266396), False),
        'ZAF': ((-25.55309193975245, 26.666126704669345), False),
    },
    'geo_contribution_2024_cross_country': {
        'BLR': ((8.14194760952201, -25.87395995350135), True),
        'BRA': ((8.435533998025875, 0.7961997362758311), False),
        'CAN': ((10.12127984763525, -12.51472669216096), False),
        'CHN': ((8.505039739793348, -25.94777917364039), False),
        'DEU': ((10.326255828135501, -5.38079683265898), True),
        'EST': ((8.955154232820902, -2.6373712332923205), False),
        'FRA': ((10.044636044344337, -2.9882854418579186), False),
        'GBR': ((10.53724354077411, -7.934856138979491), True),
        'HUN': ((8.90652899760585, -15.475419451587769), False),
        'IND': ((7.249179858660955, -3.7727154782042263), False),
        'IRN': ((7.848205670625421, -24.115850555501808), True),
        'ISR': ((10.156765307879706, -18.156871425078453), False),
        'JPN': ((10.60289776309434, 0.11715642671020277), False),
        'LAO': ((6.917670139238911, 11.111548579161898), True),
        'MNG': ((7.419814698840727, 8.668246244916567), True),
        'MOZ': ((6.046929514763946, 9.030494666584673), False),
        'NIC': ((7.068721484637518, -27.464187145293273), False),
        'NOR': ((10.56742145812332, 2.6818009813965844), False),
        'QAT': ((10.343767156785535, 10.257922875507745), False),
        'RUS': ((8.390557876291584, -29.11800719508308), True),
        'SAU': ((9.179611146463376, 0.7503153237874756), False),
        'SGP': ((10.852008220451108, 12.496500042769277), False),
        'TUR': ((9.07077203513525, -12.830306011927327), False),
        'UKR': ((6.810414369119625, 1.9228700228877855), False),
        'USA': ((10.6612432265415, -16.032634621327013), False),
        'ZAF': ((8.097353202967492, -1.815577302810027), False),
    },
}
NDIR = 24
PLACER = dict(gap_pt=1.0, margin_pt=1.0, ring_pt=3.0, assoc_pt=3.0, max_ring=2, max_ring_connector=4,
              faded_cost=1.0)


def marker_extents(ax):
    """(x, y, radius, collection index, point index) of every scatter marker in display units."""
    ppt = ax.get_figure(root=True).dpi / 72.0
    out = []
    for ci, coll in enumerate(ax.collections):
        offs = ax.transData.transform(coll.get_offsets())
        sizes, lws, ecs = coll.get_sizes(), coll.get_linewidths(), coll.get_edgecolors()
        for k, (x, y) in enumerate(offs):
            r = np.sqrt(sizes[k % len(sizes)]) / 2.0
            if len(ecs) and ecs[k % len(ecs)][3] > 0:
                r += lws[k % len(lws)] / 2.0
            out.append((x, y, r * ppt, ci, k))
    return out


def place_labels(fig, ax, texts, targets, legend, key):
    """Move each text (ha/va 'center') next to its dot; add the connectors of the previous version."""
    previous = PREVIOUS[key]
    fig.canvas.draw()
    rr = fig.canvas.get_renderer()
    ppt = fig.dpi / 72.0
    targets = np.asarray(targets, dtype=float)
    T = ax.transData.transform(targets)
    names = [t.get_text() for t in texts]
    sizes = [(b.width, b.height) for b in (t.get_window_extent(rr) for t in texts)]
    ext = marker_extents(ax)
    dots = np.array([(e[0], e[1]) for e in ext])
    dot_r = np.array([e[2] for e in ext])
    hard = np.array([ax.collections[e[3]].get_zorder() == 3 for e in ext])     # labelled (opaque) dots
    own = []
    for i in range(len(texts)):
        d = np.hypot(dots[:, 0] - T[i, 0], dots[:, 1] - T[i, 1])
        own.append(int(np.argmin(d - 1e-6 * dot_r)))
    previous = {nm: previous.get(nm, (tuple(targets[i]), False)) for i, nm in enumerate(names)}   # new labels: no preferred side
    prefer = []
    for i, nm in enumerate(names):
        v = ax.transData.transform([previous[nm][0]])[0] - T[i]
        prefer.append(np.arctan2(v[1], v[0]) if np.hypot(*v) > 1e-6 else np.nan)
    conn = [previous[nm][1] for nm in names]
    P = PLACER
    rmax = [P['max_ring_connector'] if cf else P['max_ring'] for cf in conn]
    lb, ab = legend.get_window_extent(rr), ax.get_window_extent(rr)
    cands = candidates(T, sizes, own, dots, dot_r, hard, (lb.x0, lb.y0, lb.x1, lb.y1),
                       (ab.x0, ab.y0, ab.x1, ab.y1), np.array(prefer), rmax,
                       P['gap_pt'] * ppt, P['margin_pt'] * ppt, P['ring_pt'] * ppt, ndir=NDIR,
                       faded_cost=P['faded_cost'], assoc=P['assoc_pt'] * ppt)
    choice, _, _ = solve(cands, sizes, P['margin_pt'] * ppt)
    inv = ax.transData.inverted()
    for i, t in enumerate(texts):
        t.set_position(inv.transform(cands[i][choice[i]][3:5]))
    for i, t in enumerate(texts):              # adjustText's arrow construction, same arrowprops as before
        if conn[i]:
            ax.add_patch(FancyArrowPatch(posA=t.get_position(), posB=tuple(targets[i]), patchA=t,
                                         transform=ax.transData,
                                         arrowstyle='-', color='gray', alpha=0.4, lw=0.5))
