import os
from inspiredco.critique import Critique

INSPIRED_API_KEY = os.environ.get('INSPIREDCO_API_KEY')

client = Critique(INSPIRED_API_KEY)