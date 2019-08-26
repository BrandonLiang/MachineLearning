import functions
import sys

table = functions.Spark_Function(sys.argv[1])

if len(sys.argv) > 2:
  if sys.argv[3] == "False":
    boo = False
  else:
    boo = True
  table.df_show(int(sys.argv[2]), boo)
else:
  table.df_show()
