from transformers import T5ForConditionalGeneration, RobertaTokenizer


model_name = "Salesforce/codet5-base"

konum = "weights_all_models/Salesforce_codet5-base-codexglue-sum-javascript"

# Modeli indir ve "salesforce_codet5-base-multi-sum" klasörüne kaydet
# model = T5ForConditionalGeneration.from_pretrained(model_name)
# model.save_pretrained(konum)

# Tokenizer'ı indir ve "salesforce_codet5-base-multi-sum" klasörüne kaydet
tokenizer = RobertaTokenizer.from_pretrained(model_name)
tokenizer.save_pretrained(konum)
