import numpy as np
import random
from collections import deque

MINE = -1
UNKNOWN = 0
# 숫자는 N+1

class MsEngine:
    def __init__(self, width, height, num_mines):
        self.width = width
        self.height = height
        self.num_mines = num_mines
        self.grid = np.zeros((height, width), dtype=int)
        indices = random.sample(range(width * height), num_mines)
        for i in indices:
            self.grid[i//width][i%width] = MINE
    # 칸을 여는 함수. 열린 칸을 모두 return 하고 지뢰일경우 -1, -1, -1 리턴
    def open(self, x, y):
        # OutOfRange 케어
        if not (0 <= x < self.width and 0 <= y < self.height):
            return []
        # 지뢰 밟았을 경우 밟았다고 알려줌
        if self.grid[y][x] == MINE:
            return [(-1, -1, -1)]
        # 누른 곳 또 누르면 그냥 빈 리스트 뱉음.
        if self.grid[y][x] != UNKNOWN:
            return []
        # 누른 곳의 숫자 확인
        self.grid[y][x] = self.getNum(x, y)
        # 결과에 자기 자신 넣기
        res = [(x, y, self.grid[y][x])]
        # 만약 0을 눌렀을 경우, 연쇄 반응을 BFS로 돌린다.
        if self.grid[y][x] == 1:
            # BFS용 큐
            q = deque()
            # 큐 초기화
            q.append((x,y))
            # 연쇄 반응 루프
            while q:
                top = q.popleft()
                delta = [-1, 0, 1]
                for dx in delta:
                    for dy in delta:
                        nx = top[0]+dx
                        ny = top[1]+dy
                        # OutOfBounds 케어
                        if not (0 <= nx < self.width and 0 <= ny < self.height):
                            continue
                        # 이미 열린 칸은 굳이 또 건들 필요 없음
                        if self.grid[ny][nx] == UNKNOWN:
                            self.grid[ny][nx] = self.getNum(ny, nx)
                            # 열린 모든 칸을 결과에 넣어야 함.
                            res.append((nx, ny, self.grid[ny][nx]))
                            if self.grid[ny][nx] == 1:
                                q.append((nx, ny))
        return res
    # 해당 위치에 쓰인 숫자를 찾아주는 함수
    def getNum(self, x, y):
        # 0은 모름을 나타내기 때문에 0은 1로, 1은 2로 처리함.
        res = 1
        delta = [-1, 0, 1]
        for dx in delta:
            for dy in delta:
                nx = x+dx
                ny = y+dy
                #OutOfBounds 케어
                if not (0 <= nx < self.width and 0 <= ny < self.height):
                    continue
                if self.grid[ny][nx] == MINE:
                    res = res+1
        return res
