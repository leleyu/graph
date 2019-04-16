# A C++ version of GraphSage with PyTorch

### Compile
```bash
mkdir build
cd build; cmake ..
make 
```

### Running Unsupervised example
```bash
cd build
mkdir unsupervised_cora
./unsupervised
```

### Evaluating the embeddings
```bash
cd eval_scripts
python cora_eval.py -d ../build/unsupervised_cora -l ../data/cora-label_map.json
```


