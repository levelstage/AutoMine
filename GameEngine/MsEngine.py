import numpy as np
import random
from collections import deque

MINE = -1
UNKNOWN = 0
# 숫자는 N+1

class MsEngine:
    def __init__(self, width=10, height=10, num_mines=10):
        self.width = width
        self.height = height
        self.num_mines = num_mines
        self.closed_left = width*height - num_mines
        self.grid = np.zeros((height, width), dtype=int)
        indices = random.sample(range(width * height), num_mines)
        for i in indices:
            self.grid[i//width][i%width] = MINE
    # 칸을 여는 함수. 앞의 것은 게임이 끝났는지, 뒤의 것은 지뢰를 밟았는지 여부를 return.
    def open(self, x, y):
        if not (0 <= x < self.width and 0 <= y < self.height):
            return (False, False)
        if self.grid[y][x] != UNKNOWN:
            return (False, False)
        if self.grid[y][x] == MINE:
            return (True, True)
        q = deque()
        self.grid[y][x] = self.getNum(x, y)
        self.closed_left = self.closed_left - 1
        if self.grid[y][x] == 1:
            q.append((x,y))
            while q:
                top = q.popleft()
                delta = [-1, 0, 1]
                for dx in delta:
                    for dy in delta:
                        nx = top[0]+dx
                        ny = top[1]+dy
                        if not (0 <= x < self.width and 0 <= y < self.height):
                            continue
                        if self.grid[ny][nx] == UNKNOWN:
                            self.grid[ny][nx] = self.getNum(ny, nx)
                            self.closed_left = self.closed_left - 1
                            if self.grid[ny][nx] == 1:
                                q.append((nx, ny))
        return (self.closed_left == 0, False)
    def getNum(self, x, y):
        res = 1
        delta = [-1, 0, 1]
        for dx in delta:
            for dy in delta:
                nx = x+dx
                ny = y+dy
                if not (0 <= x < self.width and 0 <= y < self.height):
                    continue
                if self.grid[ny][nx] == MINE:
                    res = res+1
        return res
