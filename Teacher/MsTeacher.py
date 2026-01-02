import numpy as np
import random
from collections import deque
from GameEngine import MsEngine

# 인공지능 모델한테 지뢰찾기의 규칙을 알려줄 선생님
class MsTeacher:
    # 생성자: 풀이를 위해 기본적으로 필요한 정보들 초기화.
    def __init__(self, width=10, height=10, num_mines=15):
        # 게임 판의 가로 길이
        self.width = width
        # 게임 판의 세로 길이
        self.height = height
        # 설치된 지뢰의 개수
        self.num_mines = num_mines
    
    # 테스트 케이스 한 판을 만드는 함수. 테스트케이스 제작이 목적이므로 지뢰를 밟을 경우 초기화 없이 다시 시도한다.
    def solve(self):
        # 만들어진 테스트 케이스 리스트
        problems = []
        answers = []
        # 밝혀지지 않은 칸의 리스트
        dark = []
        # 처음에는 모든 칸이 밝혀지지 않았으므로 전부 추가
        for i in range(self.height):
            for j in range(self.width):
                dark.append((j, i))
        # 공개된 정보들을 기록하는 그리드
        # 0은 밝혀지지 않았음을 뜻하며, 숫자가 드러날 경우 그 숫자에 1을 더한 값을 저장할 것.
        grid = np.zeros(shape=(self.height, self.width), dtype=np.int8)
        # 정답 레이블이 될 확률표 (초기값은 모두 균등함. 그럴 수밖에 없음.)
        safety = np.full(grid.shape, 1.0 - self.num_mines/(self.height *self.width), dtype=np.float32)
        # 이번 데이터 묶음에서 사용할 게임 판. (내용물은 블랙박스임)
        game = MsEngine(self.width, self.height, self.num_mines)
        
        # 메인 루프. 밝혀지지 않은 칸이 없을 때 까지 진행.
        while len(dark) > self.num_mines:

            # 열 칸을 고르는 루프
            # 닫힌 칸 중 아무 칸이나 골라서 열고, 열린 칸들을 delta에 기록한다.
            # 내부적으로는 밝혀지지 않은 칸들을 섞어서 앞에서부터 하나씩 꺼내는 식.
            pool = random.sample(dark, len(dark))
            delta = []
            for iter in pool:
                # 클릭(내부적으로는 MsEngine.open)으로 인해 새로 드러난 칸들을 저장하는 변수이다.
                res = game.open(iter[0], iter[1])
                
                # 빈 리스트(이미 열림 등)나 지뢰인 경우 패스
                if not res: continue
                if res[0] == (-1, -1, -1): continue

                delta = res
                break
            
            # 더 이상 열 곳이 없으면 종료
            if not delta: break

            # 델타를 통한 공개 정보 갱신 루프
            # cell은 (x, y, value) 꼴의 튜플이다.
            for cell in delta:
                x = cell[0]
                y = cell[1]
                value = cell[2]
                # 공개 정보 갱신
                grid[y][x] = value
                # 공개된 칸을 또 누르는 것은 불가능하므로 정답 레이블(안전도)도 0으로 바꾼다.
                safety[y][x] = 0.0
                # 밝혀졌으므로 어두운 칸 목록에서 삭제
                if (x, y) in dark:
                    dark.remove((x, y))
            
            # 만들어진 공개 정보를 문제집에 추가한다.
            problems.append(grid.copy())

            # 가장 핵심인 확률 갱신 루프
            # 확률 변동이 사라질 때 까지 계속 루프를 돈다.
            q = deque(delta)

            while q:
                # 큐에서 하나 꺼내 각 원소에 편의상 이름을 붙인다.
                top = q.popleft()
                x = top[0]
                y = top[1]
                value = top[2]-1
                
                # 우선 주변의 밝혀지지 않은 칸의 수와, 그 중에서도 완전히 안전한 칸, 지뢰 확정인 칸 수를 센다.
                tot = 0
                safe = 0
                mine = 0

                for dy in [-1, 0, 1]:
                    for dx in [-1, 0, 1]:
                        # 자기 자신은 어차피 밝혀진 칸이므로 넘긴다.
                        if dy==0 and dx ==0:
                            continue
                        nx = x + dx
                        ny = y + dy
                        # OutOfBounds 케어
                        if not (0 <= nx < self.width and 0 <= ny < self.height):
                            continue
                        # 이미 밝혀진 칸은 거른다.
                        if grid[ny][nx] > 0:
                            continue
                        tot += 1
                        
                        if safety[ny][nx] == 0.0:
                            mine += 1
                        elif safety[ny][nx] == 1.0:
                            safe += 1
                
                # 구한 안전 칸과 지뢰 수를 통해 다른 칸들의 안전도 근사치를 구한다. (한 칸만 Greedy하게 보므로 정확한 확률은 아니다.)
                # 어떤 한 칸에 지뢰가 있을 확률은 (value-mine)/(tot-safe-mine)이다.(조합을 이용하여 계산한다.)
                
                # 0으로 나누기 방지 (tot-safe-mine이 0이면 계산 불필요)
                unknowns = tot - mine - safe
                if unknowns <= 0:
                    continue

                prob_mine = (value - mine) / unknowns
                prob_calculated = 1.0 - prob_mine

                for dy in [-1, 0, 1]:
                    for dx in [-1, 0, 1]:
                        # 마찬가지로 자기 자신 패스
                        if dy==0 and dx ==0:
                            continue
                        nx = x + dx
                        ny = y + dy
                        # OutOfBounds 케어
                        if not (0 <= nx < self.width and 0 <= ny < self.height):
                            continue
                        # 여기도 마찬가지로 이미 밝혀진 칸은 거른다.
                        if grid[ny][nx] > 0:
                            continue

                        # 1. 확정된 정보(0.0, 1.0)는 무조건 덮어쓴다. (Fact > Estimate)
                        # 2. 둘 다 추측(Estimate)이라면 더 보수적인(작은) 값을 취한다.
                        # 3. 값이 '변했을 때만' 큐에 넣어서 전파한다.
                        
                        prev_prob = safety[ny][nx]
                        new_prob = prev_prob # 초기화
                        
                        # 확정 정보인 경우
                        if prob_calculated == 1.0 or prob_calculated == 0.0:
                            new_prob = prob_calculated
                        # 추측인 경우 기존보다 더 위험하면 갱신
                        elif safety[ny][nx] != 0.0 and safety[ny][nx] != 1.0:
                            if prob_calculated < safety[ny][nx]:
                                new_prob = prob_calculated
                        
                        # 값이 변했다면 갱신하고 전파
                        if new_prob != prev_prob:
                            safety[ny][nx] = new_prob
                            
                            # 만약 이번 갱신으로 '확정(Fact)'이 되었다면, 주변 숫자들을 깨워서 다시 계산시켜야 함
                            # (원래 추측이었는데 확정이 된 경우 -> 파급력이 큼)
                            if new_prob == 1.0 or new_prob == 0.0:
                                # 4중 반복문이지만 하는 수 없음.
                                for ddy in [-1, 0, 1]:
                                    for ddx in [-1, 0, 1]:
                                        # 마찬가지로 자기 자신 패스
                                        if ddy==0 and ddx==0:
                                            continue
                                        nnx = nx + ddx
                                        nny = ny + ddy
                                        # OutOfBounds 케어
                                        if not (0 <= nnx < self.width and 0 <= nny < self.height):
                                            continue
                                        # 원래 칸으로는 돌아가지 않는다.
                                        if nnx == x and nny == y:
                                            continue
                                        # 역으로 밝혀진 칸들로 가야 한다.
                                        if grid[nny][nnx] > 0:
                                            q.append((nnx, nny, grid[nny][nnx]))

            # 모든 확률 갱신이 끝나면 정답 레이블로 추가해준다.
            answers.append(safety.copy())
        return problems, answers
    
def generate_dataset(self, size):
        # 일단 예외 처리
        if size <= 0:
            return [], []
        
        problems = []
        answers = []
        
        print(f"데이터 생성 중... 목표: {size}개")
        
        # 목표 개수를 채울 때까지 계속 생성
        while len(problems) < size:
            dp, da = self.solve()
            problems.extend(dp)
            answers.extend(da)
        
        # 균등한 학습을 위한 셔플 로직
        
        # 1. 문제와 정답을 하나로 묶고
        combined = list(zip(problems, answers))
        
        # 2. 묶인 상태로 섞어야 함.
        random.shuffle(combined)
        
        # 3. 목표 개수만큼 자르면
        combined = combined[:size]
        
        # 4. 짜잔!
        problems, answers = zip(*combined)

        # 5. 리스트화해서 배출하면 끝
        problems = list(problems)
        answers = list(answers)
            
        print(f"생성 및 셔플 완료! (X: {len(problems)}, Y: {len(answers)})")
        return problems, answers