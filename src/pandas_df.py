import functions
import sys
import pandas as pd

if (len(sys.argv) >= 4):
  max_width = sys.argv[3]
else:
  max_width = False

table = functions.Pandas_Function(sys.argv[1], sys.argv[2], width = max_width)
table.df_show()
