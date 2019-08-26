import functions
import sys

table = functions.Pandas_Function(sys.argv[1], sys.argv[2], False)
table.df_show()
