using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Text;

namespace AutoMineGame;

public class AIProcessProxy
{
    // AI가 돌아갈 외부 프로세스를 담을 변수
    private Process _process;
    // 외부 프로세스 입장에서는 이쪽의 출력이 입력이고, 이쪽의 입력이 출력이므로:
    private StreamWriter _stdIn;
    private StreamReader _stdOut;
    // 실행시킬 프로세스의 이름 (python 모델 사용시 python, 다른 프로세스의 경우 .exe 파일)
    private string _fileName;
    // app.py의 주소 (python 사용할 경우 한정)
    private string _scriptPath;
    private int _height, _width, _mineCount;
    // 생성자. app.py의 주소를 받아온다.
    public AIProcessProxy(string fileName, string args = "")
    {
        _fileName = fileName;
        _scriptPath = args;
    }

    public void Start() {
        // Process.Start에 넘겨줄 인수들을 모아놓은 클래스
        ProcessStartInfo startInfo = new()
        {
            FileName = _fileName,  // exe 파일 경로
            Arguments = $"\"{_scriptPath}\"",  // 파이썬 모드일 때 app.py의 경로를 넘긴다. 공백이 있을 수 있으니 큰따옴표 붙이기.
            UseShellExecute = false,  // std io를 사용하기 위해 넣어준다.
            RedirectStandardInput = true,
            RedirectStandardOutput = true,
            CreateNoWindow = true,  // 콘솔창이 따로 안 뜨도록 설정
            StandardOutputEncoding = System.Text.Encoding.UTF8
        };
        // 아래에서 프로세스를 실행하고, 입/출력 스트림을 등록한다.
        string[] config;
        try
        {
            _process = Process.Start(startInfo);
            _stdIn = _process.StandardInput;
            _stdOut = _process.StandardOutput;
            config = _stdOut.ReadLine().Split(' ');
            _height = Convert.ToInt16(config[0]);
            _width = Convert.ToInt16(config[1]);
            _mineCount = Convert.ToInt16(config[2]);
        }
        catch(Exception e)
        {
            Debug.WriteLine("실행 실패, 에러 메시지: " + e.Message);
        }
        return;
    }

    public List<int> GetConfig()
    {
        return [_height, _width, _mineCount];
    }

    public double[,] GetPrediction(int[,] view)
{
    try
    {
        if (_process == null || _process.HasExited)
            throw new Exception("AI 프로세스가 실행 중이 아닙니다.");

        // 1. 입력 전달
        string inputString = "";
        for (int i = 0; i < _height; ++i)
        {
            for (int j = 0; j < _width; ++j)
            {
                inputString += view[i, j].ToString() + " ";
            }
            inputString += "\n";
        }
        _stdIn.Write(inputString);
        _stdIn.Flush();

        // 2. 응답 수신
        string rawResponse = _stdOut.ReadLine();
        if (rawResponse == null) throw new Exception("Python으로부터 응답이 없습니다 (null).");

        string[] response = rawResponse.Split(new[] { ' ' }, StringSplitOptions.RemoveEmptyEntries);
        
        // 3. 인덱스 오류 검증 (여기가 핵심)
        if (response.Length != _height * _width)
        {
            throw new Exception($"데이터 개수 불일치! 기대치: {_height * _width}, 실제: {response.Length}\n원문: {rawResponse}");
        }

        double[,] res = new double[_height, _width];
        for (int i = 0; i < response.Length; ++i)
        {
            // i / _width 연산 시 _width가 0이면 여기서도 터집니다.
            res[i / _width, i % _width] = double.Parse(response[i], System.Globalization.CultureInfo.InvariantCulture);
        }
        return res;
    }
    catch (Exception e)
    {
        // 콘솔 대신 파일에 에러를 기록합니다.
        System.IO.File.WriteAllText("error_log.txt", $"[Error] {e.Message}\n[StackTrace] {e.StackTrace}");
        throw; // 기록 후 원래대로 종료
    }
}

    public void Stop()
    {
        // 프로그램이 종료될 때 꼭 호출하여 프로세스를 종료해주자.
        if (_process != null && !_process.HasExited)
        {
            _process.Kill();
            _process.Dispose();
        }
    }
}