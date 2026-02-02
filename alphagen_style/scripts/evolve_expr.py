from __future__ import annotations
import argparse
import ast
import os
import sys
import random
import numpy as np
import pandas as pd
from typing import List, Optional, Tuple, Any

# Setup path
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from alphagen_style.dsl import OP_SPECS, WINDOW_CANDIDATES, ExprInfo, analyze_expr, eval_expr
from alphagen_style.evaluation import EvalConfig, evaluate_factor_panel, compute_reward, RewardConfig
from alphagen_style.masking import Node, Call, Col, Scalar

# --- Evolutionary Core ---

class GeneticOptimizer:
    def __init__(self, columns: List[str], ops: List[str], max_depth: int = 3):
        self.columns = columns
        self.ops = ops
        self.max_depth = max_depth
        self.rng = random.Random(42)

    def to_tree(self, expr: str) -> Node:
        """Parse string expression to Node tree."""
        try:
            tree = ast.parse(expr, mode='eval')
            return self._ast_to_node(tree.body)  # type: ignore
        except Exception as e:
            raise ValueError(f"Failed to parse expr: {expr}") from e

    def _ast_to_node(self, n: ast.AST) -> Node:
        if isinstance(n, ast.Call):
            name = n.func.id  # type: ignore
            args = [self._ast_to_node(a) for a in n.args]
            return Call(name, args)
        elif isinstance(n, ast.Name):
            return Col(n.id)
        elif isinstance(n, ast.Constant):
            return Scalar(float(n.value))
        elif isinstance(n, ast.UnaryOp):
            # map -x to neg(x)
            if isinstance(n.op, ast.USub):
                return Call("neg", [self._ast_to_node(n.operand)])
            # map +x to x (skip)
            if isinstance(n.op, ast.UAdd):
                return self._ast_to_node(n.operand)
        elif isinstance(n, ast.BinOp):
            # map x+y to add(x,y)
            left = self._ast_to_node(n.left)
            right = self._ast_to_node(n.right)
            if isinstance(n.op, ast.Add): return Call("add", [left, right])
            if isinstance(n.op, ast.Sub): return Call("sub", [left, right])
            if isinstance(n.op, ast.Mult): return Call("mul", [left, right])
            if isinstance(n.op, ast.Div): return Call("div", [left, right])
        
        raise ValueError(f"Unsupported AST node: {n}")

    def mutate(self, node: Node, depth: int = 0) -> Node:
        """Randomly mutate a subtree."""
        if self.rng.random() < 0.1: # 10% chance to replace this node entirely
             return self.random_tree(max_depth=max(1, self.max_depth - depth))
        
        if isinstance(node, Call):
            # Mutate arguments
            new_args = [self.mutate(a, depth + 1) for a in node.args]
            # Small chance to change op if compatible
            new_name = node.name
            if self.rng.random() < 0.2:
                candidates = [op for op, spec in OP_SPECS.items() 
                              if len(spec.arg_kinds) == len(node.args) 
                              and spec.window_arg == OP_SPECS[node.name].window_arg]
                if candidates:
                    new_name = self.rng.choice(candidates)
            return Call(new_name, new_args)
        
        elif isinstance(node, Col):
            if self.rng.random() < 0.3:
                return Col(self.rng.choice(self.columns))
            return node
            
        elif isinstance(node, Scalar):
            if self.rng.random() < 0.3:
                # Mutate value slightly or pick new window
                v = node.value
                if v in WINDOW_CANDIDATES:
                    return Scalar(float(self.rng.choice(WINDOW_CANDIDATES)))
                else:
                    return Scalar(v * self.rng.uniform(0.8, 1.2))
            return node
            
        return node
        
    def constant_node(self) -> Node:
        if self.rng.random() < 0.5:
             return Scalar(float(self.rng.choice([-2, -1, -0.5, 0.5, 1, 2])))
        return Scalar(float(self.rng.choice(WINDOW_CANDIDATES)))

    def random_tree(self, max_depth: int) -> Node:
        if max_depth <= 0 or self.rng.random() < 0.3:
            # Terminal
            if self.rng.random() < 0.8:
                return Col(self.rng.choice(self.columns))
            else:
                return self.constant_node()
        
        # Function
        op = self.rng.choice(self.ops)
        spec = OP_SPECS[op]
        args = []
        for i, kind in enumerate(spec.arg_kinds):
            if spec.window_arg == i:
                args.append(Scalar(float(self.rng.choice(WINDOW_CANDIDATES))))
            elif kind == "scalar":
                args.append(self.constant_node())
            else:
                args.append(self.random_tree(max_depth - 1))
        return Call(op, args)

    def crossover(self, p1: Node, p2: Node) -> Node:
        """Swap subtrees."""
        # Simple crossover: use p1 as base, replace a random child with p2's random subtree
        # For simplicity, we just return a new mixed tree 
        # This implementation is a placeholder for full GP crossover
        if self.rng.random() < 0.5:
            return self.mutate(p1)
        return self.mutate(p2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", required=True)
    ap.add_argument("--input", required=True, help="Input expressions file (one per line)")
    ap.add_argument("--output", required=True)
    ap.add_argument("--generations", type=int, default=5)
    ap.add_argument("--pop-size", type=int, default=50)
    args = ap.parse_args()

    # Load Panel
    df = pd.read_csv(args.panel)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df = df.set_index("timestamp")
    
    # Feature columns
    exclude_prefix = ("ret_",)
    cols = [c for c in df.columns if c != "timestamp" and not c.startswith(exclude_prefix)]
    
    # Load Initial Population
    with open(args.input, "r") as f:
        exprs = [l.strip() for l in f if l.strip()]
    
    optimizer = GeneticOptimizer(columns=cols, ops=list(OP_SPECS.keys()))
    
    population = []
    for e in exprs:
        try:
            population.append(optimizer.to_tree(e))
        except:
            pass
            
    # Evaluation function
    def evaluate(ind: Node) -> float:
        expr_str = ind.to_expr()
        try:
            # Quick calc
            # We use eval_expr + evaluate_factor_panel
            # This is heavy. In real GP, use cached results or subsampled data.
            # Using simple IC on last 1000 rows for speed
            sub_df = df.iloc[-2000:].copy() 
            res = evaluate_factor_panel(sub_df, factor_col=None, factor_expr=expr_str, target_cols=["ret_fwd_log"], 
                                        cfg=EvalConfig())
            reward = compute_reward(res, cfg=RewardConfig(trade_dd_penalty=2.0))
            return float(reward["reward"])
        except Exception as e:
            return -999.0

    print(f"Starting GP with population {len(population)} for {args.generations} gens...")
    
    for g in range(args.generations):
        # Evaluate
        scored = []
        for ind in population:
            s = evaluate(ind)
            scored.append((s, ind))
        
        scored.sort(key=lambda x: x[0], reverse=True)
        print(f"Gen {g}: Best Reward {scored[0][0]:.4f} -> {scored[0][1].to_expr()}")
        
        # Select best
        parents = [s[1] for s in scored[:10]]
        
        # Next Gen
        next_gen = parents[:] # Elitism
        while len(next_gen) < args.pop_size:
            p = random.choice(parents)
            child = optimizer.mutate(p)
            next_gen.append(child)
        
        population = next_gen
        
    # Save best
    with open(args.output, "w") as f:
        for ind in population:
            f.write(ind.to_expr() + "\n")

if __name__ == "__main__":
    main()
