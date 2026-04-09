import asyncio
import json
from app.llm_utils import LLMHybridLayer
from dotenv import load_dotenv

async def test_final():
    load_dotenv()
    layer = LLMHybridLayer()
    text = "T. Garanti Bankası ... HAVALE ... HAKAN DEMİRCİ ... FATIH SULTAN EROL ... OĞLUMUZ KAĞAN İÇİN ÇEYREK ALTIN ... TUTAR : - 250,00 TL"
    res = await layer.extract_dynamic_json([text], category='BANKING')
    print(json.dumps(res, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    asyncio.run(test_final())
