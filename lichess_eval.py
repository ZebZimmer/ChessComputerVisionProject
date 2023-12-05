import berserk
import os

session = berserk.TokenSession(os.getenv('API_TOKEN'))
client = berserk.Client(session=session)
print(session)
print(client.analysis.get_cloud_evaluation('rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1'))