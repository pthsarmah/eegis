import numpy as np
from fitness import evaluate_genome


def optimize(name, features, labels, train, validation, generations=100,
             seed=42, patience=None):
    rng = np.random.default_rng(seed)
    population = rng.integers(0, 2, size=(100, 228), dtype=np.int8)
    history = []
    best = (0., population[0].copy())
    stale = 0
    def score(genome):
        value = evaluate_genome(genome, features, labels, train, validation, name)[0]
        return value[0] if isinstance(value, tuple) else value
    for generation in range(generations):
        numeric = np.asarray([score(g) for g in population])
        order = np.argsort(numeric)
        if numeric[order[-1]] > best[0]:
            best = (float(numeric[order[-1]]), population[order[-1]].copy())
            stale = 0
        else:
            stale += 1
        history.append((generation + 1, float(numeric.max()), float(numeric.mean())))
        print(f"[ga:{name}] generation={generation + 1}/{generations} "
              f"best={numeric.max():.4f} average={numeric.mean():.4f}",
              flush=True)
        parents = population[order[-50:]]
        offspring = []
        for _ in range(50):
            a, b = parents[rng.integers(0, 50, 2)]
            mask = rng.random(228) < .5
            child = np.where(mask, a, b).copy()
            if rng.random() < .9:
                child = np.where(rng.random(228) < .5, a, b)
            child ^= (rng.random(228) < 1 / 228).astype(np.int8)
            offspring.append(child)
        population = np.vstack((population, offspring))
        keep = np.argsort([score(g) for g in population])[-100:]
        population = population[keep]
        if patience and stale >= patience:
            break
    return best, history
