from sys import argv, exit
from komod import chunk_queue

script, log_val = argv

if log_val == 'log':
    log_val = True
elif log_val == 'nolog':
    log_val = False
else:
    print("log_val must be either 'log' or 'nolog'")
    exit(1)
    
chunk_queue(log=log_val)

