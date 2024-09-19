import time
from transformers import T5ForConditionalGeneration, RobertaTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer, T5Tokenizer
import torch

konum = "weights_all_models/Salesforce_codet5p-770m-py"

# Model ve tokenizer'ı yükle
tokenizer = RobertaTokenizer.from_pretrained(konum)
model = T5ForConditionalGeneration.from_pretrained(konum)

## Özetlenecek kod girilir
code = """

write javascript function to calculate the sum of given two numbers

"""

# code = """

# Function that takes a temperature in Celsius as input from theuser and converts it to Fahrenheit.
# The program should print the converted temperature with an appropriate message.

# """


print("\n\n")

# GPU ile çalıştırma
if torch.cuda.is_available():
    model.cuda()  # Modeli GPU'ya taşı
    start_time = time.time()  # Zaman ölçümü başlar
    input_ids = tokenizer(code, return_tensors='pt').input_ids.cuda()  # GPU için input'u taşı
    outputs_gpu = model.generate(input_ids, max_new_tokens=300, 
                                 repetition_penalty=1.2,  # Tekrar cezalandırması
                                 no_repeat_ngram_size=3 # Tekrar eden n-gram'ların boyutu
                                )
    gpu_time = time.time() - start_time  # Zaman ölçümü biter
    print("\n\nGPU Result:", tokenizer.decode(outputs_gpu[0], skip_special_tokens=True))
    print(f"GPU execution time: {gpu_time:.2f} seconds")
else:
    print("CUDA is not available. GPU test skipped.")