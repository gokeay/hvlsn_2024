"""
Inference example
"""
import time

from transformers import T5ForConditionalGeneration
from transformers import RobertaTokenizer
import torch

tokenizer = RobertaTokenizer.from_pretrained("./weights_all_models/Salesforce_codet5-base-multi-sum")
model = T5ForConditionalGeneration.from_pretrained('./weights_all_models/api/saved-pretrained-kde-cpp-tm').cuda()

## çzetlenecek kod örneği
code = """

function calculateSquares(numbers) {
    const result = {};

    numbers.forEach(number => {
        if (typeof number === 'number') {
            result[number] = number * number;
        } else {
            console.warn(`${number} is not a number`);
        }
    });

    return result;
}

console.log(squares);
"""
start_time = time.time()  # Zaman ölçümü başlar

input_ids = tokenizer(code, return_tensors='pt').input_ids.cuda()



# çıktıyı özelleştirme
outputs = model.generate(
    input_ids, 
    max_new_tokens=300, 
    repetition_penalty=1.2,  # Tekrar cezalandırması
    no_repeat_ngram_size=3   # 3-gram tekrarlarını engelle
)

gpu_time = time.time() - start_time  # Zaman ölçümü biter

print("Result:", tokenizer.decode(outputs[0], skip_special_tokens=True))

print(f"GPU execution time: {gpu_time:.2f} seconds")