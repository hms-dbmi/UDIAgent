
salloc -p kempner_h100 --account=kempner_mzitnik_lab -N 1 -c 24 --mem=200G --gres=gpu:1 -t 0-23:00 -C h100
module load ncf/1.0.0-fasrc01
module load miniconda3/py39_4.11.0-linux_x64-ncf
conda activate udienv

vllm serve mims-harvard/TxAgent-T1-Llama-3.1-8B --gpu-memory-utilization 0.85 --port 8080 --host 127.0.0.1
vllm serve agenticx/UDI-VIS-Beta-v0-Llama-3.1-8B --gpu-memory-utilization 0.85 --port 8080 --host 127.0.0.1



Port forwarding in VsCode

in compute node:
- start server:

e.g. python3 -m http.server 8000 --bind 127.0.0.1
or fastapi dev ./src/udi_api.py

in login node:
- forward compute to login

e.g. ssh -L <outgoing port>:localhost:<compute port> <cluster hostname>
- compute port: matches the outgoing port from the compute node
- the outgoing port matches what you will use locally to access
- note: if port is in use it will fail with "Could not request local forwarding.", try again with a diff port.
- the cluster hostname can be determined from `hostname` in the compute

ssh -L 9090:localhost:8000 holygpu8a19404.rc.fas.harvard.edu

In ports tab of vscode, add <outgoing port>, e.g. 9090

open localhost:9090 in browser.


Errors I've run into:
- https://github.com/vllm-project/vllm/issues/13815