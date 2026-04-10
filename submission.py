"""
MATH 170 - Lab 09 Real: OOP and Einsum
Spring 2026

Instructions:
1. Implement a small Vector2D class using basic OOP.
2. Use torch.einsum for a batched tensor-matrix multiplication.
"""

import torch


class Vector2D:

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __str__(self):
        return f"Vector2D({self.x}, {self.y})"

    def add(self, other):
        return Vector2D(self.x + other.x, self.y + other.y)

    def dot(self, other):
        return self.x * other.x + self.y * other.y

def batch_right_multiply_einsum(T: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    return torch.einsum("bmn,nk->bmk", T, B)
