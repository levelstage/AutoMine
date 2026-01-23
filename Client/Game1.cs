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

    const int TILE_SIZE = 30, TILE_MARGIN = 5, SCREEN_MARGIN = 50, SECTOR_MARGIN = 100, TITLE_SECTOR_SIZE=200, GAME_WIDTH = 10, GAME_HEIGHT = 10, GAME_MINE_COUNT = 15;

    public Game1()
    {
        _graphics = new GraphicsDeviceManager(this);
        Content.RootDirectory = "Content";
        IsMouseVisible = true;
    }

    protected override void Initialize()
    {
        // 창 크기는 판 사이즈에 맞게 구성.
        _graphics.PreferredBackBufferWidth = TILE_SIZE*GAME_WIDTH*2 + TILE_MARGIN*(GAME_WIDTH-1)*2 + SCREEN_MARGIN*2 + SECTOR_MARGIN;
        _graphics.PreferredBackBufferHeight = TITLE_SECTOR_SIZE + TILE_SIZE*GAME_HEIGHT + TILE_MARGIN*(GAME_HEIGHT-1) + SCREEN_MARGIN*2;
        _graphics.ApplyChanges();

        // 새로운 게임 생성
        _board = new(GAME_WIDTH, GAME_HEIGHT, GAME_MINE_COUNT);
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
            if(currentMouse.X >= SCREEN_MARGIN || currentMouse.X <= SCREEN_MARGIN + TILE_SIZE*GAME_WIDTH + TILE_MARGIN*(GAME_WIDTH-1))
            {
                if(currentMouse.Y >= SCREEN_MARGIN + TITLE_SECTOR_SIZE || currentMouse.Y <= SCREEN_MARGIN + TITLE_SECTOR_SIZE + TILE_SIZE*GAME_HEIGHT + TILE_MARGIN*(GAME_HEIGHT-1))
                {
                    // 정확히 칸을 누른건지 체크(마진 부분을 누르지는 않았는지)
                    if((currentMouse.X - SCREEN_MARGIN) % (TILE_SIZE+TILE_MARGIN) <= TILE_SIZE && (currentMouse.Y - SCREEN_MARGIN - TITLE_SECTOR_SIZE) % (TILE_SIZE+TILE_MARGIN) <= TILE_SIZE)
                    {
                        // 이 모든걸 통과해야만 그리드 계산 후 오픈.
                        int gridX = (currentMouse.X - SCREEN_MARGIN) / (TILE_SIZE+TILE_MARGIN), girdY = (currentMouse.Y - SCREEN_MARGIN - TITLE_SECTOR_SIZE) / (TILE_SIZE+TILE_MARGIN);
                        if(_board.Open(girdY, gridX))
                        {
                            // 댁 지뢰 밟았소 처리. 일단 로그로 대체
                            Debug.WriteLine("MINE");
                        }
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
        for(int i=0; i<GAME_HEIGHT; ++i)
        {
            for(int j=0; j<GAME_WIDTH; ++j)
            {
                Color color;
                if(_board.View[i, j] == (int)BoardEnum.CLOSED || _board.View[i, j] == (int)BoardEnum.FLAG) color = Color.DarkGray;
                else color = Color.LightGray;
                _spriteBatch.Draw(
                    _pixel,
                    new Rectangle(SCREEN_MARGIN + (TILE_SIZE+TILE_MARGIN)*j, SCREEN_MARGIN + TITLE_SECTOR_SIZE + (TILE_SIZE+TILE_MARGIN)*i, TILE_SIZE, TILE_SIZE),
                    color
                );
                if(_board.View[i, j] > (int)BoardEnum.NUMBER_ZERO)
                {
                    Vector2 textSize = _font.MeasureString((_board.View[i,j]-(int)BoardEnum.NUMBER_ZERO).ToString());
                    _spriteBatch.DrawString(
                        _font, (_board.View[i,j]-(int)BoardEnum.NUMBER_ZERO).ToString(),
                        new Vector2(SCREEN_MARGIN + (TILE_SIZE+TILE_MARGIN)*j + (TILE_SIZE - textSize.X)/2, SCREEN_MARGIN + TITLE_SECTOR_SIZE + (TILE_SIZE+TILE_MARGIN)*i + (TILE_SIZE - textSize.Y)/2), 
                        Color.Black
                    );
                }
            }
        }
        _spriteBatch.End();
        base.Draw(gameTime);
    }
}
