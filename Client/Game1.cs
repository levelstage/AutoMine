using System.Collections.Generic;
using System.Diagnostics;
using Microsoft.Xna.Framework;
using Microsoft.Xna.Framework.Graphics;
using Microsoft.Xna.Framework.Input;

namespace AutoMineGame;

public class Game1 : Game
{
    private GraphicsDeviceManager _graphics;
    private SpriteBatch _spriteBatch;
    private SpriteFont _font;
    private Texture2D _pixel;
    private Board _board;
    private MouseState _prevMouseState;
    private AIProcessProxy _ai;

    const int TILE_SIZE = 30, TILE_MARGIN = 5, SCREEN_MARGIN = 50, SECTOR_MARGIN = 100, TITLE_SECTOR_SIZE=200; 
    int _gameWidth = 10, _gameHeight = 10, _gameMineCount = 10;

    double[,] _probGrid;

    public Game1()
    {
        _graphics = new GraphicsDeviceManager(this);
        Content.RootDirectory = "Content";
        IsMouseVisible = true;
    }

    protected override void Initialize()
    {
        // 사용할 AI proxy를 가져와 시동을 걸어준다.
        _ai = new AIProcessProxy("python", "C:\\Junsu\\DLStudy\\AutoMine\\AI\\app.py");
        _ai.Start();

        // 그 다음 모델에서 판의 정보를 빼온다.
        List<int> config = _ai.GetConfig();
        _gameHeight = config[0];
        _gameWidth = config[1];
        _gameMineCount = config[2];
        
        // 창 크기는 판 사이즈에 맞게 구성.
        _graphics.PreferredBackBufferWidth = TILE_SIZE*_gameWidth*2 + TILE_MARGIN*(_gameWidth-1)*2 + SCREEN_MARGIN*2 + SECTOR_MARGIN;
        _graphics.PreferredBackBufferHeight = TITLE_SECTOR_SIZE + TILE_SIZE*_gameHeight + TILE_MARGIN*(_gameHeight-1) + SCREEN_MARGIN*2;
        _graphics.ApplyChanges();

        // 새로운 게임 생성
        _board = new(_gameWidth, _gameHeight, _gameMineCount);

        base.Initialize();
    }

    protected override void LoadContent()
    {
        _spriteBatch = new SpriteBatch(GraphicsDevice);
        _font = Content.Load<SpriteFont>("FileFont");
        _pixel = new Texture2D(GraphicsDevice, 1, 1);
        _pixel.SetData([Color.White]);
    }

    protected override void Update(GameTime gameTime)
    {
        if (GamePad.GetState(PlayerIndex.One).Buttons.Back == ButtonState.Pressed || Keyboard.GetState().IsKeyDown(Keys.Escape))
            Exit();
        MouseState currentMouse = Mouse.GetState();
        // 클릭을 방금 시작한것인지 체크
        if(currentMouse.LeftButton == ButtonState.Pressed && _prevMouseState.LeftButton == ButtonState.Released)
        {
            // 플레이어블 영역 안에 있는지 체크
            if(currentMouse.X >= SCREEN_MARGIN && currentMouse.X <= SCREEN_MARGIN + TILE_SIZE*_gameWidth + TILE_MARGIN*(_gameWidth-1))
            {
                if(currentMouse.Y >= SCREEN_MARGIN + TITLE_SECTOR_SIZE && currentMouse.Y <= SCREEN_MARGIN + TITLE_SECTOR_SIZE + TILE_SIZE*_gameHeight + TILE_MARGIN*(_gameHeight-1))
                {
                    // 정확히 칸을 누른건지 체크(마진 부분을 누르지는 않았는지)
                    if((currentMouse.X - SCREEN_MARGIN) % (TILE_SIZE+TILE_MARGIN) <= TILE_SIZE && (currentMouse.Y - SCREEN_MARGIN - TITLE_SECTOR_SIZE) % (TILE_SIZE+TILE_MARGIN) <= TILE_SIZE)
                    {
                        // 이 모든걸 통과해야만 그리드 계산 후 오픈.
                        int gridX = (currentMouse.X - SCREEN_MARGIN) / (TILE_SIZE+TILE_MARGIN), girdY = (currentMouse.Y - SCREEN_MARGIN - TITLE_SECTOR_SIZE) / (TILE_SIZE+TILE_MARGIN);
                        if(_board.Open(girdY, gridX))
                        {
                            // 댁 지뢰 밟았소 처리. 일단 로그로 대체
                            _board = new Board(_gameWidth, _gameHeight, _gameMineCount);
                        }
                        _probGrid = _ai.GetPrediction(_board.View);
                    }
                }
            }
        }
        _prevMouseState = currentMouse;
        base.Update(gameTime);
    }

    protected override void Draw(GameTime gameTime)
    {
        GraphicsDevice.Clear(Color.CornflowerBlue);
        _spriteBatch.Begin();
        // --- [타이틀 영역] ---
        string title = "AUTOMINE PROJECT - AI SOLVER";
        string info = $"MAP: {_gameWidth}x{_gameHeight} | MINES: {_gameMineCount}";
        _spriteBatch.DrawString(_font, title, new Vector2(SCREEN_MARGIN, 30), Color.White);
        _spriteBatch.DrawString(_font, info, new Vector2(SCREEN_MARGIN, 70), Color.LightGray);
        _spriteBatch.DrawString(_font, "LEFT: PLAYER | RIGHT: AI PREDICTION", new Vector2(SCREEN_MARGIN, 110), Color.Yellow);
        // 좌측(플레이어 영역)을 그려준다.
        int leftMargin = SCREEN_MARGIN, topMargin = SCREEN_MARGIN + TITLE_SECTOR_SIZE;
        for(int i=0; i<_gameHeight; ++i)
        {
            for(int j=0; j<_gameWidth; ++j)
            {
                Color color;
                if(_board.View[i, j] == (int)BoardEnum.CLOSED || _board.View[i, j] == (int)BoardEnum.FLAG) color = Color.DarkGray;
                else color = Color.LightGray;
                _spriteBatch.Draw(
                    _pixel,
                    new Rectangle(leftMargin + (TILE_SIZE+TILE_MARGIN)*j, topMargin  + (TILE_SIZE+TILE_MARGIN)*i, TILE_SIZE, TILE_SIZE),
                    color
                );
                if(_board.View[i, j] > (int)BoardEnum.NUMBER_ZERO)
                {
                    Vector2 textSize = _font.MeasureString((_board.View[i,j]-(int)BoardEnum.NUMBER_ZERO).ToString());
                    _spriteBatch.DrawString(
                        _font, (_board.View[i,j]-(int)BoardEnum.NUMBER_ZERO).ToString(),
                        new Vector2(leftMargin + (TILE_SIZE+TILE_MARGIN)*j + (TILE_SIZE - textSize.X)/2, topMargin + (TILE_SIZE+TILE_MARGIN)*i + (TILE_SIZE - textSize.Y)/2), 
                        Color.Black
                    );
                }
            }
        }

        int aiLeftMargin = SCREEN_MARGIN + SECTOR_MARGIN + TILE_SIZE * _gameWidth + TILE_MARGIN * (_gameWidth - 1);
        for (int i = 0; i < _gameHeight; ++i)
        {
            for (int j = 0; j < _gameWidth; ++j)
            {
                Color color;
                bool isClosed = (_board.View[i, j] == (int)BoardEnum.CLOSED || _board.View[i, j] == (int)BoardEnum.FLAG);

                if (isClosed)
                {
                    float prob = (_probGrid != null) ? (float)_probGrid[i, j] : 0f;

                    // 3색 로직: 0(안전) -> 0.5(애매) -> 1(위험)
                    if (prob < 0.5f)
                    {
                        // 0.0 ~ 0.5 사이: 빨강(위험))에서 노랑(애매)으로
                        color = Color.Lerp(Color.Red, Color.Yellow, prob * 2f);
                    }
                    else
                    {
                        // 0.5 ~ 1.0 사이: 노랑(애매)에서 파랑(안전)으로
                        color = Color.Lerp(Color.Yellow, Color.Blue, (prob - 0.5f) * 2f);
                    }
                }
                else
                {
                    color = Color.LightGray; // 열린 칸
                }

                _spriteBatch.Draw(_pixel, new Rectangle(aiLeftMargin + (TILE_SIZE + TILE_MARGIN) * j, topMargin + (TILE_SIZE + TILE_MARGIN) * i, TILE_SIZE, TILE_SIZE), color);

                // 숫자 표시 (기존과 동일)
                if (!isClosed && _board.View[i, j] > (int)BoardEnum.NUMBER_ZERO)
                {
                    string numText = (_board.View[i, j] - (int)BoardEnum.NUMBER_ZERO).ToString();
                    Vector2 textSize = _font.MeasureString(numText);
                    _spriteBatch.DrawString(_font, numText, new Vector2(aiLeftMargin + (TILE_SIZE + TILE_MARGIN) * j + (TILE_SIZE - textSize.X) / 2, topMargin + (TILE_SIZE + TILE_MARGIN) * i + (TILE_SIZE - textSize.Y) / 2), Color.Black);
                }
            }
        }
        _spriteBatch.End();
        base.Draw(gameTime);
    }
}
