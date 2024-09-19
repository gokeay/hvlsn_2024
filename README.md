# HVLSN 2024

## Projenin Amacı
Bu projenin amacı, internetten bağımsız olarak lokal ortamda **JavaScript** kodlarını analiz edebilen **CodeT5** tabanlı bir **LLM** modeli geliştirmektir. Proje kapsamında, hazır modellerin kurulumu, çalıştırılması ve değerlendirilmesi gerçekleştirilmiştir.

Proje boyunca toplamda 3 **CodeT5** ve 1 **CodeT5+** modeli incelenmiştir. Bu modellerden üçü kod analizi yapmak için, bir tanesi ise metin girdisiyle kod oluşturmak için kullanılabilmektedir. Modeller temel görevleri yerine getirebilmekte, ancak daha iyi performans için **fine-tuning** yapılması önerilmektedir.

## Projede Kullanılan Modeller
### 1. Model: `api/saved-pretrained-kde-cpp-tm`
- **Model:** `T5ForConditionalGeneration`
- **Tokenizer:** `Salesforce/codet5-base-multi-sum` | `RobertaTokenizer`
- **Görev:** Girilen kodun analizi (**code-to-text**)
- **Link:** [CodeT5-KDE](https://github.com/tm243/CodeT5-KDE)

### 2. Model: `Salesforce/codet5-base-multi-sum`
- **Model:** `T5ForConditionalGeneration`
- **Tokenizer:** `Salesforce/codet5-base-multi-sum` | `RobertaTokenizer`
- **Görev:** Girilen kodun analizi (**code-to-text**)
- **Link:** [Hugging Face - CodeT5 Multi-Sum](https://huggingface.co/Salesforce/codet5-base-multi-sum)

### 3. Model: `Salesforce/codet5-base-codexglue-sum-javascript`
- **Model:** `T5ForConditionalGeneration`
- **Tokenizer:** `Salesforce/codet5-base` | `RobertaTokenizer`
- **Görev:** Girilen metne göre kod üretilmesi (**text-to-code**)
- **Link:** [Hugging Face - CodeT5 JavaScript](https://huggingface.co/Salesforce/codet5-base-codexglue-sum-javascript)

### 4. Model: `Salesforce/codet5p-770m-py`
- **Model:** `T5ForConditionalGeneration`
- **Tokenizer:** `Salesforce/codet5p-770m-py` | `RobertaTokenizer`
- **Görev:** Girilen metne göre kod üretilmesi (**text-to-code**)
- **Link:** [Hugging Face - CodeT5 Python](https://huggingface.co/Salesforce/codet5p-770m-py)

## Projenin Bilgisayara Kurulumu
**Github Linki:** [HVLSN 2024 - Github](https://github.com/gokeay/hvlsn_2024)

### Adımlar:
1. Proje klonlanır ve proje dizinine girilir:
    ```bash
    git clone https://github.com/gokeay/hvlsn_2024
    cd hvlsn_2024
    ```

2. Sanal ortam oluşturulur ve etkinleştirilir (isteğe bağlı):
    ```bash
    python -m venv myenv
    .\myenv\Scripts\activate
    ```

3. Gerekli bağımlılıklar indirilir:
    ```bash
    pip install -r requirements.txt
    ```

4. Her model, ayrı dosyalar içerisinde yer alır. `main.py`, `main_2.py`, `main_3.py` ve `main_4.py` sırasıyla **model_1**, **model_2**, **model_3** ve **model_4**'ü çalıştırır.

## Notlar
1. Her model dosyasında, örnek kodlar yorum satırları içerisinde bulunabilir.
2. **model_1** ağırlıkları, ilgili Github bağlantısı aracılığıyla indirilmelidir.
3. Modeli javascript özelinde eğitmek için 60k javascript ve özet içeriği: https://huggingface.co/datasets/google/code_x_glue_ct_code_to_text/viewer/javascript?row=7
