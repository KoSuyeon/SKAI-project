from transformers import AutoTokenizer, AutoModel

# 모델 리스트
models = [
    "intfloat/multilingual-e5-large",
    "intfloat/multilingual-e5-base",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
]

# 각 모델에 대해 tokenizer, model 로딩 및 정보 출력
for model_name in models:
    print(f"\nModel: {model_name}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    print(" - Max token length:", tokenizer.model_max_length)
    print(" - Hidden size:", model.config.hidden_size)
