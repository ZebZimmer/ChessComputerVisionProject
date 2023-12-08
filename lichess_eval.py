import berserk
import os
from stockfish import Stockfish
import requests

# session = berserk.TokenSession(os.getenv('API_TOKEN'))
# client = berserk.Client(session=session)
# print(session)
# print(client.analysis.get_cloud_evaluation('8/8/3k4/1P6/2K3P1/8/8/8 w - - 0 1'))

# Stockfish.set_fen_position('8/3p4/7Q/2k5/7Q/2K5/8/8 w - - 0 20')
# print(stockfish.is_fen_valid("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"))
# print(stockfish.get_best_move())



def send_FEN_get_move(FEN_str):
    url = f"https://stockfish.online/api/stockfish.php?fen={FEN_str}&depth=5&mode=bestmove"
    print(url)
    response = requests.get(url)
    print(response.text)
    # session = berserk.TokenSession(os.getenv('API_TOKEN'))
    # client = berserk.Client(session=session)
    # print(client.analysis.get_cloud_evaluation(fen = FEN_str))