import requests


def send_FEN_get_move(FEN_str):
    url = f"https://stockfish.online/api/stockfish.php?fen={FEN_str}&depth=5&mode=bestmove"
    print(url)
    response = requests.get(url)
    print(response.text)