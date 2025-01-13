# run_experiments.py
import importlib
import numpy as np
from tqdm import tqdm
import sys

def run_multiple_trials(game_module, num_trials=1000):
    """Run multiple trials of a Hanabi variant."""
    scores = []
    print(f"\nRunning {num_trials} games of {game_module.__name__}")
    
    for _ in tqdm(range(num_trials)):
        try:
            score = game_module.run_game()
            scores.append(score)
        except Exception as e:
            print(f"\nError in trial: {e}")
            continue
    
    # Calculate statistics
    scores = np.array(scores)
    stats = {
        'version': game_module.__name__,
        'trials': num_trials,
        'average': np.mean(scores),
        'std': np.std(scores),
        'min': np.min(scores),
        'max': np.max(scores),
        'median': np.median(scores)
    }
    
    print(f"\nResults for {stats['version']}:")
    print(f"Trials: {stats['trials']}")
    print(f"Average score: {stats['average']:.2f} ± {stats['std']:.2f}")
    print(f"Min/Max: {stats['min']}/{stats['max']}")
    print(f"Median: {stats['median']}")
    
    return stats

if __name__ == "__main__":
    # Default to vanilla if no argument provided
    version = sys.argv[1] if len(sys.argv) > 1 else "hanabi_vanilla"
    num_trials = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
    
    # Import specified version
    try:
        game = importlib.import_module(version)
        stats = run_multiple_trials(game, num_trials)
    except ImportError:
        print(f"Could not find Hanabi version: {version}")
        sys.exit(1)
