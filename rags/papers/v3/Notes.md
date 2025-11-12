# reference
+ https://microsoft.github.io/graphrag/get_started/

# prepare data
```bash
python /app/rags/papers/v3/pdf2txt.py
```

# init folder
```bash
graphrag init --root /app/rags/papers/v3
```

# edit
+ API key in .env
+ add to .gitignore
+ model name in settings.yaml

# indexing
```bash
graphrag index --root /app/rags/papers/v3
```

# retrieve
```bash
graphrag query \
    --root /app/rags/papers/v3 \
    --method global \
    --query "What are the key areas that medicine focuses on to ensure well-being?"
```
