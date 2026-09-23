



import os
import sys
import time
import re
import gc
import json
import asyncio
# torch / transformers are imported lazily, inside the functions that need them.
# The Ollama serving path never touches them, and they are NOT in requirements.txt
# (only requirements-finetune.txt), so an eager import here makes `import server`
# fail with ModuleNotFoundError on a standard install. See _try_torch() below.


def _try_torch():
    """Return the torch module, or None if it is not installed.

    Used by the optional local-HuggingFace path. The Ollama path does not need
    torch, so its absence must never be fatal.
    """
    try:
        import torch
        return torch
    except ImportError:
        return None

try:
    import ollama as _ollama_lib
except ImportError:
    _ollama_lib = None

# ============================================================
# 配置区 — 从 config.yaml 加载（如缺失则用内置默认值）
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

try:
    from config_loader import cfg as _cfg
    if _cfg and _cfg.ollama_base_url:
        os.environ.setdefault("OLLAMA_HOST", _cfg.ollama_base_url)
except Exception as _e:
    print(f"  ⚠️  config_loader 加载失败: {_e}，使用内置默认值")
    _cfg = None

# 生成参数（从 config.yaml 读取）
MAX_NEW_TOKENS    = _cfg.max_new_tokens    if _cfg else 4096
TEMPERATURE       = _cfg.temperature       if _cfg else 0.65
TOP_P             = _cfg.top_p             if _cfg else 0.95
REPETITION_PENALTY= _cfg.repetition_penalty if _cfg else 1.05
DO_SAMPLE         = _cfg.do_sample         if _cfg else True
QUANT_MODE        = _cfg.quant_mode        if _cfg else "4bit"
DEFAULT_THINK_MODE= _cfg.default_think_mode if _cfg else False

# 将 `Model Networking` 库添加到搜索路径
sys.path.append(os.path.join(BASE_DIR, "Model Networking"))
try:
    from web_search import WebResearcher
except ImportError:
    WebResearcher = None


def print_banner(model_name=""):
    print("\n" + "=" * 64)
    print(f"  🚀 Universal HF Model Runner (FP16/4bit for RTX 4080)")
    if model_name:
        print(f"  📦 当前加载: {model_name}")
    print("=" * 64)

def select_model_interactively():
    """扫描当前目录下所有包含 config.json 的文件夹或 .gguf 文件并让用户选择"""
    print("\n  🔍 正在扫描可用模型...")
    available_models = []
    
    # 查找当前目录及其一级子目录中的 config.json 或直接的 .gguf 文件
    for item in os.listdir(BASE_DIR):
        item_path = os.path.join(BASE_DIR, item)
        
        # 支持 GGUF 单文件模型
        if os.path.isfile(item_path) and item.lower().endswith('.gguf'):
            available_models.append((item_path, item, "gguf"))
            continue
            
        # 支持旧的 Transformers 文件夹格式
        if os.path.isdir(item_path):
            config_path = os.path.join(item_path, "config.json")
            if os.path.exists(config_path):
                # 尝试读取模型真实名称
                model_name = item
                try:
                    with open(config_path, "r", encoding="utf-8") as f:
                        config_data = json.load(f)
                    if "_name_or_path" in config_data:
                        model_name = f"{item} ({config_data['_name_or_path']})"
                except Exception:
                    pass
                available_models.append((item_path, model_name, "hf"))
                
    if not available_models:
        print("  ❌ 错误: 在当前目录下没有找到任何 .gguf 文件或包含 config.json 的模型文件夹！")
        sys.exit(1)
        
    print("  📋 请选择要加载的模型 (输入序号):")
    for i, (path, name, mtype) in enumerate(available_models):
        print(f"     [{i+1}] {name} [{mtype.upper()}]")
        
    while True:
        try:
            choice = input("\n  > ").strip()
            idx = int(choice) - 1
            if 0 <= idx < len(available_models):
                selected_path, selected_name, selected_type = available_models[idx]
                print(f"  ✅ 已选择: {selected_name}")
                return selected_path, selected_name, selected_type
            else:
                print("  ⚠️ 输入无效，请输入左侧的序号数字。")
        except ValueError:
            print("  ⚠️ 请输入数字！")
        except (KeyboardInterrupt, EOFError):
            print("\n  👋 退出程序。")
            sys.exit(0)


def apply_speed_optimizations():
    """针对本地 HuggingFace 推理路径的优化 (Ollama 路径无需 torch, 直接跳过)"""
    torch = _try_torch()
    if torch is None:
        return
    if torch.cuda.is_available():
        # TF32 加速矩阵运算 (Ampere/Ada GPU)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.cuda.set_device(0)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")
    # 全局禁用梯度计算
    torch.set_grad_enabled(False)


def load_model_and_tokenizer(model_path, model_type="hf"):
    """加载模型: 兼容 HF 格式和 GGUF 格式"""

    if model_type == "gguf":
        if _ollama_lib is None:
            print("  ❌ 错误: 未安装 ollama Python 包!")
            print("  请运行: pip install ollama")
            sys.exit(1)

        # 从文件名推导 Ollama 模型名称 (去掉量化后缀 Q4_K_M 等)
        stem = os.path.splitext(os.path.basename(model_path))[0]
        # 匹配量化后缀，支持 - 和 _ 两种分隔符
        ollama_name = re.sub(r'[_-]Q\d[A-Z0-9_]*$', '', stem, flags=re.IGNORECASE).lower().replace('_', '-')

        print(f"  ⏳ 正在连接 Ollama (模型: {ollama_name})...")
        try:
            models_resp = _ollama_lib.list()
            registered = [m.model.split(':')[0] for m in models_resp.models]
            if ollama_name not in registered:
                print(f"  ❌ Ollama 中找不到模型 '{ollama_name}'")
                print(f"  请在项目目录下运行: ollama create {ollama_name} -f Modelfile")
                sys.exit(1)
        except Exception as e:
            print(f"  ❌ 无法连接到 Ollama 服务: {e}")
            print("  请确认 Ollama 已安装并正在运行 (系统托盘图标)")
            sys.exit(1)

        print(f"  ✅ Ollama 就绪! 模型: {ollama_name} (RTX 4080 全速 GPU 加速)")
        return ollama_name, "ollama"

    # 以下是原生的 HuggingFace 加载逻辑 (需要 torch / transformers)
    torch = _try_torch()
    if torch is None:
        print("  ❌ 本地 HuggingFace 推理需要 torch / transformers, 但未安装。")
        print("     请运行: pip install -r requirements-finetune.txt")
        print("     (若只使用 Ollama 推理, 无需安装。)")
        sys.exit(1)
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

    if not torch.cuda.is_available():
        print("  ❌ CUDA 不可用!")
        sys.exit(1)

    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem = torch.cuda.get_device_properties(0).total_memory
    gpu_mem_gb = gpu_mem / 1024**3
    print(f"  🖥️  GPU: {gpu_name} ({gpu_mem_gb:.1f} GB)")

    # 加载 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        padding_side="left",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ---- 智能评估和构建加载参数 ----
    # 我们先看看 config.json 获取模型的大致情况
    try:
        config_path = os.path.join(model_path, "config.json")
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
    except Exception:
        cfg = {}

    # 简单启发式：通过 hidden_size 和 num_hidden_layers 猜测模型大小
    # 比如 8B 模型通常层数>=32，隐藏层大小>=4096
    is_large_model = cfg.get("num_hidden_layers", 32) >= 32 and cfg.get("hidden_size", 4096) >= 4096
    
    quant_mode = QUANT_MODE
    if not is_large_model and quant_mode in ["4bit", "8bit"]:
        print("  💡 检测到模型较小，将尝试使用全速无量化(FP16)模式加载...")
        quant_mode = "fp16"
        
    print(f"  ⏳ 正在加载模型 (模式: {quant_mode})...")

    load_kwargs = {
        "dtype": torch.float16,
        "trust_remote_code": True,
        "low_cpu_mem_usage": True,
    }

    # 给 GPU 预留 1GB 以下的安全余量，避免爆显存
    gpu_safe_mem = max(1, int(gpu_mem_gb) - 1)

    if quant_mode == "fp16":
        load_kwargs["device_map"] = "auto"
        load_kwargs["max_memory"] = {
            0: f"{gpu_safe_mem}GiB",
            "cpu": "16GiB",   
        }
    elif quant_mode == "4bit":
        load_kwargs["device_map"] = "auto"
        load_kwargs["max_memory"] = {0: f"{gpu_safe_mem}GiB", "cpu": "4GiB"}
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,  # RTX 4080
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
    elif quant_mode == "8bit":
        load_kwargs["device_map"] = "auto"
        load_kwargs["max_memory"] = {0: f"{gpu_safe_mem}GiB", "cpu": "4GiB"}
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=6.0,
            llm_int8_enable_fp32_cpu_offload=False,
        )

    # 尝试不同的 Attention 实现
    for attn_impl in ["flash_attention_2", "sdpa", None]:
        try:
            if attn_impl:
                load_kwargs["attn_implementation"] = attn_impl
            else:
                load_kwargs.pop("attn_implementation", None)

            model = AutoModelForCausalLM.from_pretrained(model_path, **load_kwargs)

            if attn_impl:
                attn_name = "Flash Attention 2" if attn_impl == "flash_attention_2" else "SDPA"
                print(f"  ⚡ {attn_name} Attention 已启用")
            break
        except Exception as e:
            if attn_impl is None:
                raise
            continue

    model.eval()

    # 检查设备分布
    if hasattr(model, 'hf_device_map'):
        device_info = model.hf_device_map
        gpu_layers = sum(1 for v in device_info.values() if str(v) == '0' or str(v).isdigit())
        cpu_layers = sum(1 for v in device_info.values() if v == 'cpu')
        total_layers = len(device_info)
        if cpu_layers > 0:
            print(f"  📊 GPU: {gpu_layers} 层 | CPU: {cpu_layers} 层 | 共 {total_layers} 层")
            print(f"  ℹ️  部分层在 CPU 上 (FP16 无量化, 但 GPU 显存不够放全部)")
            print(f"  ℹ️  GPU 上的层依然以全速运算, 整体仍比 bitsandbytes 快")
        else:
            print(f"  ✅ 全部 {total_layers} 层都在 GPU 上!")

    vram_used = torch.cuda.memory_allocated(0) / 1024**3
    print(f"  ✅ 模型已加载! GPU 显存: {vram_used:.1f}/{gpu_mem_gb:.1f} GB")

    # 预热
    print("  ⏳ 预热中...", end="", flush=True)
    try:
        warmup_inputs = tokenizer("Hello", return_tensors="pt", padding=True)
        warmup_ids = warmup_inputs["input_ids"].to(model.device)
        warmup_mask = warmup_inputs["attention_mask"].to(model.device)
        with torch.inference_mode():
            _ = model.generate(
                warmup_ids,
                attention_mask=warmup_mask,
                max_new_tokens=5,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        with torch.inference_mode():
            _ = model.generate(
                warmup_ids,
                attention_mask=warmup_mask,
                max_new_tokens=10,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        del warmup_ids, warmup_mask, warmup_inputs
        torch.cuda.empty_cache()
        gc.collect()
        print(" ✅ 完成")
    except Exception as e:
        print(f" ⚠️ {e}")

    return model, tokenizer


def build_prompt(tokenizer, messages, think_mode=True, is_resume=False):
    is_last_assistant = False
    if messages:
        last = messages[-1]
        is_last_assistant = getattr(last, "get", lambda k: last.get(k))("role") == "assistant" if not isinstance(last, dict) else last.get("role") == "assistant"
        
    if is_resume and is_last_assistant:
        # Format history without the last assistant message
        base_text = tokenizer.apply_chat_template(
            messages[:-1], tokenize=False, add_generation_prompt=True,
        )
        # Append the partial assistant response
        input_text = base_text + messages[-1]['content']
        return input_text

    if think_mode:
        input_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
    else:
        input_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False,
        )
        input_text += "<｜Assistant｜><think>\n</think>\n\n"
    return input_text


def extract_response(raw_text, think_mode=True):
    if think_mode:
        think_match = re.search(r'<think>(.*?)</think>(.*)', raw_text, re.DOTALL)
        if think_match:
            return think_match.group(1).strip(), think_match.group(2).strip()
        if '<think>' in raw_text and '</think>' not in raw_text:
            return raw_text.replace('<think>', '').strip(), ""
        return "", raw_text.strip()
    else:
        clean = re.sub(r'<think>.*?</think>', '', raw_text, flags=re.DOTALL).strip()
        return "", clean if clean else raw_text.strip()


class ThinkingAwareStreamer:
    """
    自定义 Streamer: 解决 HuggingFace 原生 TextStreamer 吞字/等空格导致“无法逐字输出”的问题，并正确处理思考标签。
    """
    def __init__(self, tokenizer, think_mode=True, show_thinking=True, skip_prompt=True, **kwargs):
        self.tokenizer = tokenizer
        self.think_mode = think_mode
        self.show_thinking = show_thinking
        self.skip_prompt = skip_prompt
        self.is_first_chunk = True
        
        self.token_cache = []
        self.print_len = 0
        self.all_text = ""
        self.phase = "waiting"
        self.printed_text_len = 0
        self.answer_header_printed = False
        
        # 收集需要过滤的特殊 token 文本 (除了 <think>/</think>)
        self._special_strings = set()
        if hasattr(tokenizer, 'all_special_tokens'):
            for t in tokenizer.all_special_tokens:
                if t not in ('<think>', '</think>'):
                    self._special_strings.add(t)

    def _clean(self, text):
        """清理特殊标签, 保留 <think>/</think>"""
        for s in self._special_strings:
            text = text.replace(s, '')
        return text

    def put(self, value):
        if self.is_first_chunk:
            self.is_first_chunk = False
            if self.skip_prompt:
                return
                
        if len(value.shape) > 1:
            value = value[0]
        self.token_cache.extend(value.tolist())
        text = self.tokenizer.decode(self.token_cache, skip_special_tokens=False)
        
        # 处理不完整的 utf-8 字节产生的未知字符(通常是 \ufffd)
        if text.endswith('\ufffd'): 
            return
            
        new_text = text[self.print_len:]
        self.print_len = len(text)
        self.on_finalized_text(new_text, stream_end=False)

    def end(self):
        if self.token_cache:
            text = self.tokenizer.decode(self.token_cache, skip_special_tokens=False)
            new_text = text[self.print_len:]
            if new_text and not new_text.endswith('\ufffd'):
                self.on_finalized_text(new_text, stream_end=True)
            else:
                self.on_finalized_text("", stream_end=True)
                
        self.token_cache = []
        self.print_len = 0

    def on_finalized_text(self, text: str, stream_end: bool = False):
        self.all_text += text

        # 普通模型模式，或者不需要抓取思考过程时，直接一字不差地输出给控制台
        if not self.think_mode:
            clean_text = self._clean(text)
            print(clean_text, end="", flush=True)
            if stream_end:
                print(flush=True)
            return

        # ================= 思考模式逻辑 =================
        new_chunk = self.all_text[self.printed_text_len:]
        if not new_chunk and not stream_end:
            return

        if self.phase == "waiting":
            self.phase = "thinking"
            if self.show_thinking:
                print("\n  💭 [思考中...]\n", flush=True)

        if self.phase == "thinking":
            if '</think>' in self.all_text:
                self.phase = "answering"
                
                # 打印到 </think> 为止的思考过程残余
                think_end_idx = self.all_text.find('</think>')
                unprinted_think = self.all_text[self.printed_text_len : think_end_idx]
                if self.show_thinking and unprinted_think:
                    print(self._clean(unprinted_think.replace('<think>', '')), end="", flush=True)
                
                if self.show_thinking:
                    print(f"\n\n  💭 [思考结束]\n", flush=True)
                
                if not self.answer_header_printed:
                    print("  🤖 回答: ", end="", flush=True)
                    self.answer_header_printed = True
                
                # 打印 </think> 之后的内容
                answer_part = self.all_text[think_end_idx + len('</think>') :]
                if answer_part:
                    print(self._clean(answer_part), end="", flush=True)
                
                self.printed_text_len = len(self.all_text)
            else:
                if self.show_thinking:
                    clean_chunk = self._clean(new_chunk.replace('<think>', ''))
                    if clean_chunk:
                        print(clean_chunk, end="", flush=True)
                self.printed_text_len = len(self.all_text)

        elif self.phase == "answering":
            if new_chunk:
                print(self._clean(new_chunk), end="", flush=True)
            self.printed_text_len = len(self.all_text)

        if stream_end:
            if not self.answer_header_printed:
                if self.phase == "thinking" and self.show_thinking:
                    print(f"\n\n  💭 [思考结束]\n", flush=True)
                print("  🤖 回答: ", end="", flush=True)
            print(flush=True)


def generate_response(model, tokenizer, messages, think_mode=True,
                      show_thinking=True, stream=True):
                      
    # ====== Ollama (GGUF via Ollama) 生成逻辑 ======
    if tokenizer == "ollama":
        import ollama as _ol
        max_tokens = MAX_NEW_TOKENS + (512 if think_mode else 0)

        start_time = time.time()
        all_text = ""
        phase = "waiting"  # waiting -> thinking -> answering
        final_answer = ""
        final_think = ""

        print("\n  ⏳ 正在生成...", end="", flush=True)
        if not think_mode and stream:
            print("\n  🤖 AI回答: ", end="", flush=True)

        try:
            response = _ol.chat(
                model=model,
                messages=messages,
                stream=stream,
                think=bool(think_mode),
                options={
                    "temperature": TEMPERATURE if DO_SAMPLE else 0.0,
                    "top_p": TOP_P if DO_SAMPLE else 1.0,
                    "repeat_penalty": REPETITION_PENALTY,
                    "num_predict": max_tokens,
                }
            )

            if stream:
                for chunk in response:
                    piece = chunk['message']['content']
                    if not piece:
                        continue
                    all_text += piece

                    if not think_mode:
                        print(piece, end="", flush=True)
                        final_answer += piece
                        continue

                    if phase == "waiting":
                        if '<think>' in all_text:
                            phase = "thinking"
                            if show_thinking:
                                print("\n  💭 [思考中...]\n", flush=True)
                                parts = all_text.split('<think>', 1)
                                if len(parts) > 1:
                                    print(parts[1], end="", flush=True)
                                    final_think += parts[1]

                    elif phase == "thinking":
                        if '</think>' in all_text:
                            phase = "answering"
                            if show_thinking:
                                print(f"\n\n  💭 [思考结束]\n", flush=True)
                            print("  🤖 回答: ", end="", flush=True)
                            parts = all_text.split('</think>', 1)
                            if len(parts) > 1:
                                print(parts[1], end="", flush=True)
                                final_answer += parts[1]
                        else:
                            if show_thinking:
                                print(piece, end="", flush=True)
                            final_think += piece

                    elif phase == "answering":
                        print(piece, end="", flush=True)
                        final_answer += piece
            else:
                raw = response['message']['content']
                all_text = raw
                if think_mode:
                    think_match = re.search(r'<think>(.*?)</think>(.*)', raw, re.DOTALL)
                    if think_match:
                        final_think = think_match.group(1).strip()
                        final_answer = think_match.group(2).strip()
                    else:
                        final_answer = raw.replace('<think>', '').strip()
                else:
                    final_answer = raw

        except Exception as e:
            print(f"\n ⚠️ Ollama 推理发生错误: {e}")
            return ""

        print()
        elapsed = time.time() - start_time
        print(f"  📊 Ollama GPU 生成 | {elapsed:.1f}s")
        return final_answer if final_answer else all_text


    # ====== 以下是 HuggingFace (Transformers/BitsAndBytes) 的生成逻辑 ======
    torch = _try_torch()
    if torch is None:
        raise RuntimeError(
            "本地 HuggingFace 生成需要 torch, 但未安装。"
            "请运行 pip install -r requirements-finetune.txt, 或改用 Ollama 推理 (tokenizer=='ollama')。"
        )

    input_text = build_prompt(tokenizer, messages, think_mode=think_mode)
    inputs = tokenizer(input_text, return_tensors="pt", padding=True)
    input_ids = inputs["input_ids"].to(model.device)
    attention_mask = inputs["attention_mask"].to(model.device)

    max_tokens = MAX_NEW_TOKENS + (512 if think_mode else 0)

    gen_kwargs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "max_new_tokens": max_tokens,
        "pad_token_id": tokenizer.pad_token_id,
        "use_cache": True,
        "cache_implementation": "static", # 新增加速配置
    }

    if DO_SAMPLE:
        gen_kwargs.update({
            "do_sample": True,
            "temperature": TEMPERATURE,
            "top_p": TOP_P,
            "repetition_penalty": REPETITION_PENALTY,
        })
    else:
        gen_kwargs["do_sample"] = False
        gen_kwargs["repetition_penalty"] = REPETITION_PENALTY

    start_time = time.time()

    if stream:
        # 无论是否是 thinking 都使用我们修复维度 bug 后的 Streamer
        streamer = ThinkingAwareStreamer(tokenizer, think_mode=think_mode,
                                        show_thinking=show_thinking)
        if not think_mode:
            print("\n  🤖 AI回答: ", end="", flush=True)
        gen_kwargs["streamer"] = streamer
        with torch.inference_mode():
            outputs = model.generate(**gen_kwargs)
    else:
        print("\n  ⏳ 正在生成...", end="", flush=True)
        with torch.inference_mode():
            outputs = model.generate(**gen_kwargs)

    elapsed = time.time() - start_time

    generated_ids = outputs[0][input_ids.shape[1]:]
    # think_mode 时保留 <think> 标签以便 extract_response 正确解析
    raw_response = tokenizer.decode(generated_ids, skip_special_tokens=not think_mode)
    thinking, answer = extract_response(raw_response, think_mode=think_mode)

    if not stream:
        if think_mode and thinking and show_thinking:
            print(f"\n\n  💭 [思考过程]\n  {thinking}\n  💭 [思考结束]\n")
        print(f"  🤖 DeepSeek: {answer}")

    num_tokens = len(generated_ids)
    tokens_per_sec = num_tokens / elapsed if elapsed > 0 else 0
    print(f"\n  📊 生成 {num_tokens} tokens | {elapsed:.1f}s | {tokens_per_sec:.1f} tokens/s")

    del input_ids, attention_mask, inputs, outputs, generated_ids
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return answer if answer else raw_response


def single_query_mode(model, tokenizer):
    print("  思考模式? (1=开启, 2=关闭, 默认2): ", end="")
    think_choice = input().strip()
    think_mode = think_choice == "1"
    query = input("  > ").strip()
    if not query:
        return
    messages = [{"role": "user", "content": query}]
    generate_response(model, tokenizer, messages, think_mode=think_mode)


def interactive_chat_mode(model, tokenizer, model_name=""):
    torch = _try_torch()
    print("\n  💬 对话模式 | /help 查看命令 | /quit 退出")
    
    # 联网模块开启查询
    use_web = False
    if WebResearcher is not None:
        choice = input("  🌐 开启新版实时联网搜索功能? (y/N): ").strip().lower()
        use_web = choice == 'y'

    # 根据模型名字智能猜测是否需要默认开启思考模式 (带 R1/DeepSeek/Think 的通常需要)
    is_reasoning_model = any(keyword in model_name.lower() for keyword in ["r1", "deepseek", "reason", "think"])
    think_mode = is_reasoning_model
    show_thinking = True

    if think_mode:
        print("  ⚡ 思考机制 [已开启]")
    else:
        print("  💬 普通对话 [已开启]")
        
    if use_web:
        print("  🌐 实时联网 [已开启]")
        web_researcher = WebResearcher(_cfg) if _cfg else WebResearcher()

    messages = []
    system_prompt = None

    while True:
        try:
            print()
            user_input = input("  👤 你: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n\n  👋 再见!")
            break

        if not user_input:
            continue

        if user_input.startswith("/"):
            cmd_parts = user_input.lower().split()
            cmd = cmd_parts[0]

            if cmd in ("/quit", "/exit"):
                print("\n  👋 再见!")
                break
            elif cmd == "/clear":
                messages = []
                if torch is not None and torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()
                print("  🗑️  对话历史已清空。")
                if system_prompt:
                    print(f"  ℹ️  系统提示词保留: {system_prompt[:50]}...")
                continue
            elif cmd == "/system":
                system_prompt = user_input[len("/system"):].strip() or None
                print(f"  ✅ 系统提示词: {'已设置' if system_prompt else '已清除'}")
                continue
            elif cmd == "/think":
                if len(cmd_parts) > 1:
                    think_mode = cmd_parts[1] in ("on", "1", "true", "yes")
                else:
                    think_mode = not think_mode
                print(f"  思考模式: {'✅ 开启' if think_mode else '⚡ 关闭'}")
                continue
            elif cmd == "/show_think":
                if len(cmd_parts) > 1:
                    show_thinking = cmd_parts[1] in ("on", "1", "true", "yes")
                else:
                    show_thinking = not show_thinking
                print(f"  显示思考: {'✅ 开启' if show_thinking else '❌ 关闭'}")
                continue
            elif cmd == "/tokens":
                global MAX_NEW_TOKENS
                if len(cmd_parts) > 1:
                    try:
                        new_val = int(cmd_parts[1])
                        if 64 <= new_val <= 8192:
                            MAX_NEW_TOKENS = new_val
                            print(f"  最大 tokens: {MAX_NEW_TOKENS}")
                        else:
                            print("  范围: 64 - 8192")
                    except ValueError:
                        print("  用法: /tokens 512")
                else:
                    print(f"  当前最大 tokens: {MAX_NEW_TOKENS}")
                continue
            elif cmd == "/history":
                if not messages:
                    print("  ℹ️  对话历史为空。")
                else:
                    print(f"  📜 对话历史 ({len(messages)} 条消息):")
                    for i, msg in enumerate(messages):
                        role = "👤 用户" if msg["role"] == "user" else "🤖 AI"
                        print(f"     {i+1}. [{role}] {msg['content'][:80]}...")
                continue
            elif cmd == "/status":
                print("  📊 当前设置:")
                print(f"     🧠 思考模式:  {'✅ 开启' if think_mode else '⚡ 关闭'}")
                print(f"     👁️  显示思考:  {'✅ 是' if show_thinking else '❌ 否'}")
                print(f"     📝 最大 tokens: {MAX_NEW_TOKENS}")
                print(f"     ⚡ 量化模式:  {QUANT_MODE}")
                print(f"     💬 历史:      {len(messages)} 条消息")
                if torch is not None and torch.cuda.is_available():
                    vram = torch.cuda.memory_allocated(0) / 1024**3
                    print(f"     💾 GPU 显存:  {vram:.1f} GB")
                continue
            elif cmd == "/help":
                print("  /quit /clear /system /think on|off /show_think on|off")
                print("  /tokens <N> /history /status")
                continue
            else:
                print(f"  ⚠️  未知命令: {cmd}, /help 查看帮助")
                continue

        full_messages = []
        if system_prompt:
            full_messages.append({"role": "system", "content": system_prompt})
        full_messages.extend(messages)
        full_messages.append({"role": "user", "content": user_input})

        if use_web:
            augmented_messages, sources = asyncio.run(web_researcher.prepare(full_messages))
            if sources:
                print(f"  🌐 已读取 {len(sources)} 个来源")
            response = generate_response(
                model, tokenizer, augmented_messages,
                think_mode=think_mode, show_thinking=show_thinking,
            )
        else:
            response = generate_response(
                model, tokenizer, full_messages,
                think_mode=think_mode, show_thinking=show_thinking,
            )

        messages.append({"role": "user", "content": user_input})
        messages.append({"role": "assistant", "content": response})

        total_chars = sum(len(m["content"]) for m in messages)
        if total_chars > 10000:
            print(f"\n  ⚠️  对话较长 ({total_chars} 字符), 建议 /clear")


def main():
    model_path, model_name, model_type = select_model_interactively()
    print_banner(model_name)
    apply_speed_optimizations()

    if model_type == "hf":
        for f in ["config.json", "tokenizer.json"]:
            if not os.path.exists(os.path.join(model_path, f)):
                print(f"  ❌ 缺少文件: {f} (请确保 {model_path} 中有完整的模型文件)")
                sys.exit(1)

    model, tokenizer = load_model_and_tokenizer(model_path, model_type)

    print("  模式: 1=对话(默认) 2=单次查询")
    try:
        choice = input("  > ").strip()
    except (KeyboardInterrupt, EOFError):
        return

    if choice == "2":
        single_query_mode(model, tokenizer)
    else:
        interactive_chat_mode(model, tokenizer, model_name=model_name)


if __name__ == "__main__":
    main()
