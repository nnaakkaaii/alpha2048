#!/usr/bin/env python3
"""Enhanced testing script with batch processing support."""
import argparse
import os
from typing import Dict, Any, List, Optional
import time

import torch
import numpy as np
from tqdm import tqdm

from pkg.fields import Game, Action
from pkg.networks import ConvDQN
from pkg.utils import encode_board


def test_batch(
    model_path: str,
    num_games: int = 100,
    device: Optional[torch.device] = None,
    verbose: bool = True,
    batch_size: int = 1,  # Number of parallel games
    render_games: int = 0,  # Number of games to render
    save_stats: bool = False,
    stats_path: str = "test_stats.npz",
) -> Dict[str, Any]:
    """Test the trained model with batch processing support.
    
    Args:
        model_path: Path to the trained model
        num_games: Number of games to test
        device: Device to run on (None for auto)
        verbose: Print progress
        batch_size: Number of games to run in parallel
        render_games: Number of games to render (from the beginning)
        save_stats: Save detailed statistics
        stats_path: Path to save statistics
        
    Returns:
        Dictionary with test results
    """
    # Auto-select device if not provided
    if device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    
    if verbose:
        print(f"Testing on device: {device}")
        print(f"Batch size: {batch_size}")
        print(f"Number of games: {num_games}")

    # Load model
    model = ConvDQN().to(device)
    if os.path.exists(model_path):
        state_dict = torch.load(model_path, map_location=device, weights_only=False)
        model.load_state_dict(state_dict)
        if verbose:
            print(f"Model loaded from {model_path}")
    else:
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    model.eval()

    # Statistics collectors
    all_scores = []
    all_max_tiles = []
    all_move_counts = []
    tile_distribution = {}
    action_counts = {0: 0, 1: 0, 2: 0, 3: 0}  # Up, Down, Left, Right
    game_lengths = []
    
    # Timing
    start_time = time.time()
    
    # Progress bar
    pbar = tqdm(total=num_games, disable=not verbose, desc="Testing")
    
    games_completed = 0
    while games_completed < num_games:
        # Determine batch size for this iteration
        current_batch_size = min(batch_size, num_games - games_completed)
        
        # Initialize games
        games = [Game() for _ in range(current_batch_size)]
        states = []
        active_games = list(range(current_batch_size))
        move_counts = [0] * current_batch_size
        
        # Reset all games
        for i, game in enumerate(games):
            board = game.reset()
            state = encode_board(board, device)
            states.append(state)
        
        # Play games
        while active_games:
            # Process active games
            batch_states = []
            batch_indices = []
            
            for idx in active_games:
                game = games[idx]
                legal_actions = [a.value for a in game.get_legal_actions()]
                
                if not legal_actions:
                    # Game over
                    continue
                
                batch_states.append(states[idx])
                batch_indices.append(idx)
            
            if not batch_states:
                break
            
            # Get actions from model (batch inference)
            with torch.no_grad():
                batch_tensor = torch.stack(batch_states)
                q_values = model(batch_tensor)
            
            # Execute actions
            new_active_games = []
            for i, idx in enumerate(batch_indices):
                game = games[idx]
                legal_actions = [a.value for a in game.get_legal_actions()]
                
                if not legal_actions:
                    continue
                
                # Select best legal action
                q_vals = q_values[i].cpu().numpy()
                legal_q_values = [(a, q_vals[a]) for a in legal_actions]
                best_action = max(legal_q_values, key=lambda x: x[1])[0]
                
                # Execute action
                _, done = game.step(Action(best_action))
                move_counts[idx] += 1
                action_counts[best_action] += 1
                
                # Render if requested
                if games_completed + idx < render_games and verbose:
                    print(f"\nGame {games_completed + idx + 1}, Move {move_counts[idx]}")
                    print(f"Action: {['Up', 'Down', 'Left', 'Right'][best_action]}")
                    print(f"Score: {game.score}, Max tile: {game.get_max_tile()}")
                    game.render()
                
                if not done:
                    # Update state
                    board = game.board.copy()
                    states[idx] = encode_board(board, device)
                    new_active_games.append(idx)
            
            active_games = new_active_games
        
        # Collect statistics
        for i in range(current_batch_size):
            game = games[i]
            score = game.score
            max_tile = game.get_max_tile()
            
            all_scores.append(score)
            all_max_tiles.append(max_tile)
            all_move_counts.append(move_counts[i])
            game_lengths.append(move_counts[i])
            
            # Update tile distribution
            if max_tile not in tile_distribution:
                tile_distribution[max_tile] = 0
            tile_distribution[max_tile] += 1
        
        games_completed += current_batch_size
        pbar.update(current_batch_size)
    
    pbar.close()
    
    # Calculate statistics
    elapsed_time = time.time() - start_time
    games_per_second = num_games / elapsed_time
    
    results = {
        "num_games": num_games,
        "mean_score": np.mean(all_scores),
        "std_score": np.std(all_scores),
        "max_score": np.max(all_scores),
        "min_score": np.min(all_scores),
        "median_score": np.median(all_scores),
        "mean_max_tile": np.mean(all_max_tiles),
        "std_max_tile": np.std(all_max_tiles),
        "best_tile": np.max(all_max_tiles),
        "mean_moves": np.mean(all_move_counts),
        "std_moves": np.std(all_move_counts),
        "tile_distribution": tile_distribution,
        "action_distribution": action_counts,
        "elapsed_time": elapsed_time,
        "games_per_second": games_per_second,
        "scores": all_scores,
        "max_tiles": all_max_tiles,
        "move_counts": all_move_counts,
    }
    
    # Calculate percentiles
    percentiles = [25, 50, 75, 90, 95, 99]
    for p in percentiles:
        results[f"score_p{p}"] = np.percentile(all_scores, p)
        results[f"tile_p{p}"] = np.percentile(all_max_tiles, p)
    
    # Calculate tile achievement rates
    tile_targets = [256, 512, 1024, 2048, 4096, 8192]
    for target in tile_targets:
        achieved = sum(1 for t in all_max_tiles if t >= target)
        results[f"achieved_{target}"] = achieved
        results[f"rate_{target}"] = achieved / num_games
    
    if verbose:
        print("\n" + "="*60)
        print("TEST RESULTS")
        print("="*60)
        print(f"Games played: {num_games}")
        print(f"Time elapsed: {elapsed_time:.1f}s ({games_per_second:.1f} games/s)")
        print(f"\nScore Statistics:")
        print(f"  Mean: {results['mean_score']:.1f} ± {results['std_score']:.1f}")
        print(f"  Median: {results['median_score']:.1f}")
        print(f"  Max: {results['max_score']:.1f}")
        print(f"  Min: {results['min_score']:.1f}")
        print(f"  95th percentile: {results['score_p95']:.1f}")
        
        print(f"\nMax Tile Statistics:")
        print(f"  Mean: {results['mean_max_tile']:.1f} ± {results['std_max_tile']:.1f}")
        print(f"  Best: {results['best_tile']}")
        
        print(f"\nTile Achievement Rates:")
        for target in tile_targets:
            if results[f"achieved_{target}"] > 0:
                print(f"  ≥{target:5d}: {results[f'achieved_{target}']:4d}/{num_games} "
                      f"({results[f'rate_{target}']*100:5.1f}%)")
        
        print(f"\nTile Distribution:")
        sorted_tiles = sorted(tile_distribution.keys())
        for tile in sorted_tiles:
            count = tile_distribution[tile]
            percentage = (count / num_games) * 100
            print(f"  {tile:5d}: {count:4d} ({percentage:5.1f}%)")
        
        print(f"\nAction Distribution:")
        total_actions = sum(action_counts.values())
        for action, count in action_counts.items():
            action_name = ['Up', 'Down', 'Left', 'Right'][action]
            percentage = (count / total_actions) * 100 if total_actions > 0 else 0
            print(f"  {action_name:5s}: {count:6d} ({percentage:5.1f}%)")
        
        print(f"\nMove Statistics:")
        print(f"  Mean moves per game: {results['mean_moves']:.1f} ± {results['std_moves']:.1f}")
    
    # Save statistics if requested
    if save_stats:
        np.savez(
            stats_path,
            **results
        )
        if verbose:
            print(f"\nStatistics saved to {stats_path}")
    
    return results


def test(
    model_path: str,
    num_games: int = 100,
    device: Optional[torch.device] = None,
    verbose: bool = True,
) -> Dict[str, float]:
    """Simple test function for backward compatibility."""
    results = test_batch(
        model_path=model_path,
        num_games=num_games,
        device=device,
        verbose=verbose,
        batch_size=1,
        render_games=0,
        save_stats=False,
    )
    
    return {
        "avg_score": results["mean_score"],
        "max_tile": results["best_tile"],
        "achieved_2048": results["rate_2048"],
    }


def main():
    parser = argparse.ArgumentParser(
        description="Test trained 2048 DQN agent with enhanced features"
    )
    
    parser.add_argument("model_path", type=str, help="Path to the trained model")
    parser.add_argument("--num-games", type=int, default=100, help="Number of games to test")
    parser.add_argument("--device", type=str, default="auto", help="Device: cpu, cuda, mps, or auto")
    parser.add_argument("--quiet", action="store_true", help="Suppress output")
    parser.add_argument("--batch-size", type=int, default=1, help="Number of parallel games")
    parser.add_argument("--render", type=int, default=0, help="Number of games to render")
    parser.add_argument("--save-stats", action="store_true", help="Save detailed statistics")
    parser.add_argument("--stats-path", type=str, default="test_stats.npz", help="Path for statistics")
    
    args = parser.parse_args()
    
    # Select device
    if args.device == "auto":
        device = None
    else:
        device = torch.device(args.device)
    
    # Run tests
    results = test_batch(
        model_path=args.model_path,
        num_games=args.num_games,
        device=device,
        verbose=not args.quiet,
        batch_size=args.batch_size,
        render_games=args.render,
        save_stats=args.save_stats,
        stats_path=args.stats_path,
    )
    
    # Return code based on performance
    if results["rate_2048"] >= 0.5:  # 50% success rate for 2048
        return 0
    else:
        return 1


if __name__ == "__main__":
    exit(main())