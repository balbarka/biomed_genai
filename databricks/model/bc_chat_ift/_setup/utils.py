import json
from typing import List

def write_jsonl_by_line(responses: List, outfile: str) -> None:
    # Write to jsonl line by line
    with open(outfile, 'a+') as out:
        for r in responses and len(set(r.values()).intersection({None,'None','null'}))==0:
            if r:
                jout = json.dumps(r) + '\n'
                out.write(jout)

