from audio_analyzer import cli


def test_no_args_noninteractive_prints_help(monkeypatch, capsys):
    monkeypatch.setattr(cli.sys.stdin, "isatty", lambda: False)
    cli.main([])
    out = capsys.readouterr().out
    assert "Audio-Analyzer" in out
    assert "--system-audio" in out


def test_interactive_menu_can_exit(monkeypatch, capsys):
    answers = iter(["0"])
    monkeypatch.setattr("builtins.input", lambda _prompt="": next(answers))
    cli.run_interactive_menu()
    out = capsys.readouterr().out
    assert "Audio Analyzer" in out
    assert "PA-Einmessmodus" in out


def test_interactive_menu_dispatches_live(monkeypatch):
    answers = iter(["3", "0"])
    commands = []
    monkeypatch.setattr("builtins.input", lambda _prompt="": next(answers))
    monkeypatch.setattr(cli, "_run_argv", lambda argv: commands.append(argv))
    cli.run_interactive_menu()
    assert commands == [["--live"]]


def test_interactive_menu_file_with_spaces(monkeypatch):
    answers = iter(["1", "/tmp/My Song.wav", "0"])
    commands = []
    monkeypatch.setattr("builtins.input", lambda _prompt="": next(answers))
    monkeypatch.setattr(cli, "_run_argv", lambda argv: commands.append(argv))
    cli.run_interactive_menu()
    assert commands == [["--", "/tmp/My Song.wav"]]
