using System;
using System.Linq;
using System.Collections.Generic;

namespace AutoMineGame;

public class Board
{
    // 판의 너비, 높이, 지뢰의 수
    public  readonly int Width, Height, MineCount;
    // 지뢰와 숫자를 저장할 그리드
    public int[,] Grid { get; private set; }
    // 그래픽 처리기 및 엔진에게 보여줄 현재 판의 상태
    public int[,] View { get; private set; }

    public Board(int width, int height, int mineCount)
    {
        // 파라미터 세팅
        Width = width;
        Height = height;
        MineCount = mineCount;

        // 지뢰 칸 랜덤 뽑기
        int totalTiles = Width * Height;
        Grid = new int[height, width];
        View = new int[height, width];
        Random rnd = new Random();

        var mineIndices = Enumerable.Range(0, totalTiles)
                                    .OrderBy(x => rnd.Next())
                                    .Take(mineCount)
                                    .ToList();

        // 뽑힌 인덱스를 (x, y)로 변환해서 지뢰 심기
        foreach (int index in mineIndices)
        {
            int y = index / Width;  // 몫은 Y (행)
            int x = index % Width;  // 나머지는 X (열)
            Grid[y, x] = (int)BoardEnum.MINE;  // 지뢰로 설정.
        }
        // 게임 시작부터 끝까지 절대 변하지 않을 Grid 초기화
        for (int i=0; i<height; ++i)
        {
            for (int j=0; j<width; ++j)
            {
                if(Grid[i, j] == (int)BoardEnum.MINE) continue;
                int mineAround = (int)BoardEnum.NUMBER_ZERO;
                for(int ny=i-1; ny<=i+1; ++ny)
                {
                    if(ny < 0 || ny >= height) continue;
                    for(int nx=j-1; nx<=j+1; ++nx)
                    {
                        if(nx < 0 || nx >= width) continue;
                        if(Grid[ny, nx] == (int)BoardEnum.MINE) ++mineAround;
                    }
                }
                Grid[i, j] = mineAround;
            }
        }
    }

    public bool Open(int y, int x)
    {
        // 지뢰를 밟았을 경우 true, 아닐 경우 칸을 열고 false를 return.
        if(Grid[y, x] == (int)BoardEnum.MINE) return true;
        View[y, x] = Grid[y, x];
        // 만약 0을 눌렀을 경우, 0 기준으로 8칸을 연쇄적으로 지우는 BFS 탐색을 개시한다.
        Queue<int> q = new();
        if(Grid[y, x] == (int)BoardEnum.NUMBER_ZERO) q.Enqueue(y*Width + x);
        // 연쇄 반응 BFS 로직.
        while(q.Count > 0)
        {
            int nx = q.Peek() % Width, ny = q.Dequeue() / Width;
            for(int i=ny-1; i<=ny+1; ++i)
            {
                if(i<0 || i>=Height) continue;
                for(int j=nx-1; j<=nx+1; ++j)
                {
                    if(j<0 || j>=Width) continue;
                    // 이미 열린 칸이라면 무시하고 지나간다.
                    if(View[i, j] == Grid[i, j]) continue;
                    View[i, j] = Grid[i, j];
                    if(Grid[i, j] == (int)BoardEnum.NUMBER_ZERO) q.Enqueue(i*Width + j);
                }
            }
        }
        return false;
    }

    public void ToggleFlag(int y, int x)
    {
        if(View[y, x] == (int)BoardEnum.CLOSED) View[y, x] = (int)BoardEnum.FLAG;
        else if(View[y, x] == (int)BoardEnum.FLAG) View[y, x] = (int)BoardEnum.CLOSED;
        return;
    }    
}