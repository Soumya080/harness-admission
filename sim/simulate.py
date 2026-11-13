from scipy.stats import binomtest, pairs 
def greedy(pairs):
     pairs = list(pairs)
     return sum(c for c, _ in pairs) > sum(i for _, i in pairs), len(pairs)
 
def mcnemar_fixed(pairs, alpha=0.05):
    
    pairs = list(pairs)
    wins = sum(1 for c, i in pairs if c > i)
    losses = sum(1 for c, i in pairs if c < i)
    if wins + losses == 0:
        return False, len(pairs)
    p = binomtest(wins, wins+losses, 0.5, alternative="greater").pvalue
    return p < alpha, len(pairs)



def one_run(rng, edits, start, gate, n_dev=40, regime="controlled"):
    dev = rng.choice(DEV_IDS, n_dev, replace=False)
    inc, commits, cost = start, [], 0
    for cand in rng.permutation(edits):
        def out(h, i):
            return O[h][i] if regime == "controlled" else S[h][i][rng.integers(5)]
        pairs = ((out(cand, i), out(inc, i)) for i in rng.permutation(dev))
        ok, used = gate(pairs)
        cost += used
        if ok:
            commits.append((inc, cand))
            inc = cand
    return commits, inc, cost