import torch
from transformers import LogitsProcessor, LogitsProcessorList, AutoModelForCausalLM, AutoTokenizer
import re

# ==========================================
# 核心組件：強制奇偶隱寫處理器
# ==========================================
class ParityLogitsProcessor(LogitsProcessor):
    """
    強制隱寫處理器
    原理：使用 -inf 強制封殺不符合奇偶性的一半詞彙表
    保證 Alice 生成的 Token ID 100% 準確地攜帶秘密
    """
    def __init__(self, secret_bits):
        self.secret_bits = secret_bits
        self.bit_idx = 0

    def __call__(self, input_ids, scores):
        if self.bit_idx >= len(self.secret_bits):
            return scores

        current_bit = self.secret_bits[self.bit_idx]
        vocab_size = scores.shape[-1]
        
        is_odd = (torch.arange(vocab_size, device=scores.device) % 2) != 0
        
        if current_bit == 0:
            scores[:, is_odd] = -float("inf")
        else:
            scores[:, ~is_odd] = -float("inf")

        self.bit_idx += 1
        return scores

# ==========================================
# 工具函數
# ==========================================
def int_to_bits(n, num_bits=32):
    """將整數轉換為 32 位元列表"""
    return [int(b) for b in format(n, f'0{num_bits}b')]

def bits_to_int(bits):
    """將位元列表轉換回整數"""
    if len(bits) < 32:
        print(f"⚠️ 警告：位元長度不足 {len(bits)}/32")
        return 0
    return int("".join(str(b) for b in bits[:32]), 2)

# ==========================================
# 文本清理器
# ==========================================
class TextCleaner:
    """清理生成文本中的問題字符"""
    
    PROBLEMATIC_CHARS = [
        '\ue800', '\ue600', '\uf8ff',  # Unicode 私用區
        'ʾ', 'ʿ', 'ʼ', 'ʻ',             # 異常撇號變體
        '【', '】', '「', '」', '『', '』',  # CJK 括號
        '________',                     # 底線佔位符
    ]
    
    @staticmethod
    def clean(text: str) -> str:
        """清理問題字符並規範化空格"""
        cleaned = text
        for char in TextCleaner.PROBLEMATIC_CHARS:
            cleaned = cleaned.replace(char, '')
        cleaned = re.sub(r'\s+', ' ', cleaned)
        cleaned = cleaned.strip()
        return cleaned
    
    @staticmethod
    def truncate_to_reasonable_length(text: str, original_prompt: str) -> str:
        """
        智能截斷文本，保留描述性部分，移除敘事性延伸
        """
        # 提取續寫部分
        if text.startswith(original_prompt):
            continuation = text[len(original_prompt):].strip()
        else:
            continuation = text
        
        # 按句子分割
        sentences = re.split(r'([.!?])', continuation)
        
        # 敘事標記（表示從描述轉為故事）
        narrative_markers = [
            'being led', 'then they', 'suddenly', 'never to', 
            'given into', 'he said', 'she said', 'they were',
            'i saw', 'we found', 'you can see'
        ]
        
        # 重組句子
        result_sentences = []
        for i in range(0, len(sentences), 2):
            if i + 1 < len(sentences):
                sentence = sentences[i] + sentences[i+1]
            else:
                sentence = sentences[i]
            
            # 檢查是否包含敘事標記
            if any(marker in sentence.lower() for marker in narrative_markers):
                break
            
            result_sentences.append(sentence)
            
            # 最多保留 2-3 個描述性句子
            if len(result_sentences) >= 2:
                break
        
        truncated = original_prompt + ' ' + ''.join(result_sentences)
        return truncated.strip()

# ==========================================
# 主題檢查器
# ==========================================
class TopicChecker:
    """檢查生成文本是否保持原始主題"""
    
    TOPIC_KEYWORDS = {
        'mountain': ['mountain', 'peak', 'summit', 'cliff', 'slope', 'alpine', 'snow', 'valley', 'ridge'],
        'painting': ['painting', 'art', 'canvas', 'artwork', 'landscape', 'color', 'scene', 'image'],
        'strawberry': ['strawberry', 'berry', 'fruit', 'sweet', 'red', 'ripe', 'fresh', 'juicy'],
        'cookie': ['cookie', 'baked', 'sweet', 'dessert', 'chocolate', 'dough', 'treat'],
        'smell': ['smell', 'aroma', 'scent', 'fragrance'],
        'taste': ['taste', 'flavor', 'delicious', 'savory'],
    }
    
    @staticmethod
    def extract_main_topics(prompt: str) -> set:
        """從 prompt 提取主要主題"""
        prompt_lower = prompt.lower()
        found_topics = set()
        
        for topic, keywords in TopicChecker.TOPIC_KEYWORDS.items():
            if any(kw in prompt_lower for kw in keywords):
                found_topics.add(topic)
        
        return found_topics
    
    @staticmethod
    def check_topic_adherence(prompt: str, generated: str) -> tuple[bool, float]:
        """
        檢查主題保持度
        返回 (是否相關, 相關度分數)
        """
        main_topics = TopicChecker.extract_main_topics(prompt)
        
        if not main_topics:
            return True, 1.0
        
        generated_lower = generated.lower()
        relevant_count = 0
        
        for topic in main_topics:
            keywords = TopicChecker.TOPIC_KEYWORDS[topic]
            for kw in keywords:
                if kw in generated_lower:
                    relevant_count += 1
        
        relevance_score = relevant_count / len(main_topics) if main_topics else 0
        is_relevant = relevance_score > 0
        
        return is_relevant, relevance_score

# ==========================================
# 品質檢查器
# ==========================================
class QualityChecker:
    """檢查生成文本的品質（寬鬆版，只攔截致命問題）"""
    
    CRITICAL_PATTERNS = [
        'ersatz', 'erythroid', 'erythrocyte',  # 亂用的專業詞
        "ʼs what", '________',                  # 格式問題
    ]
    
    @staticmethod
    def check(text: str, original_prompt: str) -> tuple[bool, str]:
        """
        品質檢查
        返回 (是否通過, 原因說明)
        """
        lower_text = text.lower()
        
        # 1. 檢查致命的垃圾詞
        for pattern in QualityChecker.CRITICAL_PATTERNS:
            if pattern.lower() in lower_text:
                return False, f"包含不良模式: {pattern}"
        
        # 2. 檢查嚴重重複（放寬到 25%）
        words = text.split()
        if len(words) > 20:
            unique_ratio = len(set(words)) / len(words)
            if unique_ratio < 0.25:
                return False, f"嚴重重複: {unique_ratio:.2%}"
        
        # 3. 主題相關性檢查
        is_relevant, score = TopicChecker.check_topic_adherence(original_prompt, text)
        if not is_relevant:
            return False, f"主題完全偏離"
        
        return True, f"可接受（主題: {score:.2f}）"

# ==========================================
# 對外介面：TextStegoSystem
# ==========================================
class TextStegoSystem:
    """
    文本隱寫系統
    
    功能：
    - alice_encode: 將 32 位元金鑰隱藏在生成的文本中
    - bob_decode: 從 token IDs 中提取隱藏的金鑰
    """
    
    def __init__(self, model_name="gpt2"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)
        self.num_bits_to_hide = 32
        self.quality_checker = QualityChecker()
        self.cleaner = TextCleaner()

    def _create_descriptive_prompt(self, prompt: str) -> str:
        """
        創建強描述性 prompt，引導模型生成視覺描述而非敘事
        """
        clean_prompt = prompt.strip()
        clean_prompt = self.cleaner.clean(clean_prompt)
        
        # 使用 "The scene shows" 強制生成靜態視覺描述
        if clean_prompt.endswith(('.', '!', '?')):
            return clean_prompt + " The scene shows"
        else:
            return clean_prompt + ". The scene shows"

    def alice_encode(self, prompt: str, secret_key_int: int, max_retries: int = 12) -> tuple[str, list[int]]:
        """
        Alice 編碼函數：將秘密金鑰隱藏在生成的文本中
        
        Args:
            prompt: 原始提示文本
            secret_key_int: 要隱藏的 32 位元整數金鑰
            max_retries: 最大重試次數
            
        Returns:
            (隱寫文本, token IDs 列表)
        """
        secret_bits = int_to_bits(secret_key_int, self.num_bits_to_hide)
        total_bits_len = len(secret_bits)
        
        best_text = None
        best_tokens = None
        best_score = -1
        
        for attempt in range(max_retries):
            # 動態調整生成參數
            temperature = 0.6 + (attempt * 0.03)
            top_p = 0.85 + (attempt * 0.01)
            
            # 創建奇偶性處理器
            processor = ParityLogitsProcessor(secret_bits)
            logits_processor = LogitsProcessorList([processor])
            
            # 準備 prompt
            descriptive_prompt = self._create_descriptive_prompt(prompt)
            inputs = self.tokenizer(descriptive_prompt, return_tensors="pt").to(self.device)
            input_len = inputs.input_ids.shape[1]
            
            # 生成文本
            outputs = self.model.generate(
                **inputs,
                min_new_tokens=total_bits_len + 2,
                max_new_tokens=total_bits_len + 18,
                
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                top_k=40,
                
                logits_processor=logits_processor,
                pad_token_id=self.tokenizer.eos_token_id,
                no_repeat_ngram_size=2,
                repetition_penalty=1.3
            )
            
            stego_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            generated_token_ids = outputs[0][input_len:].tolist()
            
            # 品質檢查
            is_ok, reason = self.quality_checker.check(stego_text, prompt)
            
            if is_ok:
                # 清理並智能截斷
                cleaned_text = self.cleaner.clean(stego_text)
                truncated_text = self.cleaner.truncate_to_reasonable_length(cleaned_text, prompt)
                
                print(f"✅ 第 {attempt + 1} 次嘗試成功 - {reason}")
                print(f"   (截斷前: {len(cleaned_text.split())} 詞 → 截斷後: {len(truncated_text.split())} 詞)")
                
                return truncated_text, generated_token_ids
            else:
                print(f"⚠️ 第 {attempt + 1} 次嘗試失敗: {reason}")
                
                # 保存最佳結果（基於主題相關度）
                _, score = TopicChecker.check_topic_adherence(prompt, stego_text)
                if score > best_score:
                    best_score = score
                    best_text = stego_text
                    best_tokens = generated_token_ids
        
        # 所有嘗試都失敗，返回最佳結果
        print(f"⚠️ {max_retries} 次嘗試後返回最佳結果（主題相關度: {best_score:.2f}）")
        cleaned_best = self.cleaner.clean(best_text)
        truncated_best = self.cleaner.truncate_to_reasonable_length(cleaned_best, prompt)
        return truncated_best, best_tokens

    def bob_decode(self, generated_token_ids: list[int]) -> int:
        """
        Bob 解碼函數：從 token IDs 中提取隱藏的金鑰
        
        Args:
            generated_token_ids: Alice 生成的 token ID 列表
            
        Returns:
            解碼出的整數金鑰
        """
        num_bits_to_read = self.num_bits_to_hide
        
        if len(generated_token_ids) < num_bits_to_read:
            print(f"❌ [Bob] Token 列表太短，無法解碼 (需要: {num_bits_to_read}, 實際: {len(generated_token_ids)})")
            return 0

        # 從每個 token ID 提取奇偶位元
        extracted_bits = []
        for i in range(num_bits_to_read):
            token_id = generated_token_ids[i]
            extracted_bits.append(token_id % 2)

        return bits_to_int(extracted_bits)


# ==========================================
# 使用範例
# ==========================================
if __name__ == "__main__":
    system = TextStegoSystem()
    
    test_prompts = [
        "A beautiful painting of a mountain.",
        "The taste of a ripe strawberry was a burst of sweetness.",
        "The smell of freshly baked cookies filled the house."
    ]
    
    secret_key = 42
    
    print("=" * 70)
    print("文本隱寫系統測試")
    print("=" * 70)
    
    for i, prompt in enumerate(test_prompts, 1):
        print(f"\n【測試 {i}】")
        print(f"原文: {prompt}")
        print("-" * 70)
        
        # Alice 編碼
        stego_text, token_ids = system.alice_encode(prompt, secret_key)
        print(f"\n最終隱寫文本:\n{stego_text}\n")
        
        # 驗證主題保持
        is_relevant, score = TopicChecker.check_topic_adherence(prompt, stego_text)
        print(f"主題保持: {'✅' if is_relevant else '❌'} (相關度: {score:.2f})")
        
        # Bob 解碼
        decoded_key = system.bob_decode(token_ids)
        print(f"金鑰驗證: {'✅' if decoded_key == secret_key else '❌'} (原始: {secret_key}, 解碼: {decoded_key})")
        print("=" * 70)