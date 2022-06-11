import transformers


def get_tokenizer_and_model(pretrained_model_name) -> tuple:

    def _get_gogamza_bart():
        tokenizer = transformers.PreTrainedTokenizerFast.from_pretrained(
            pretrained_model_name,
        )
        model = transformers.BartForConditionalGeneration.from_pretrained(
            pretrained_model_name,
        )

        return tokenizer, model

    def _get_skt_kogpt2() -> tuple:
        ## Load tokenizer and model.
        tokenizer = transformers.PreTrainedTokenizerFast.from_pretrained(
            pretrained_model_name,
            bos_token="</s>",
            eos_token="</s>", 
            unk_token="<unk>",
            pad_token="<pad>", 
            mask_token="<mask>",
        )
        model = transformers.GPT2LMHeadModel.from_pretrained(
            pretrained_model_name,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            torch_dtype="auto",
        )
        return tokenizer, model

    def _get_skt_kogpt_trinity() -> tuple:
        ## Load tokenizer and model.
        tokenizer = transformers.PreTrainedTokenizerFast.from_pretrained(
            pretrained_model_name,
            bos_token="</s>",
            eos_token="</s>", 
            unk_token="<unk>",
            pad_token="<pad>", 
            mask_token="<mask>",
        )
        model = transformers.GPT2LMHeadModel.from_pretrained(
            pretrained_model_name,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            torch_dtype="auto",
        )
        return tokenizer, model

    def _get_kakao_kogpt() -> tuple:
        ## Load tokenizer and model.
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            pretrained_model_name,
            revision="KoGPT6B-ryan1.5b-float16",
            bos_token="</s>",
            eos_token="</s>", 
            unk_token="<unk>",
            pad_token="<pad>", 
            mask_token="<mask>",
        )
        model = transformers.AutoModelForCausalLM.from_pretrained(
            pretrained_model_name,
            revision="KoGPT6B-ryan1.5b-float16",
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            torch_dtype="auto",
            low_cpu_mem_usage=True,
        )
        return tokenizer, model

    ## Get one.
    return {
        "gogamza/kobart-base-v1": _get_gogamza_bart,
        "skt/kogpt2-base-v2": _get_skt_kogpt2,
        "skt/ko-gpt-trinity-1.2B-v0.5": _get_skt_kogpt_trinity,
        "kakaobrain/kogpt": _get_kakao_kogpt,
    }[pretrained_model_name]()
