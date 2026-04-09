import os
import cv2
import numpy as np
from loguru import logger
from paddleocr import PaddleOCR

# Sophisticated import for PPStructure to handle version 2.9+ (V3) and older versions
try:
    # Try the newest V3 class first (PaddleOCR 2.9+)
    from paddleocr import PPStructureV3 as PPStructure
except ImportError:
    try:
        from paddleocr import PPStructure
    except ImportError:
        try:
            # Fallback to sub-module for some specific distributions
            from paddleocr.paddleocr import PPStructure
        except ImportError:
            PPStructure = None

from .preprocessing import ImagePreprocessor

class DocumentProcessor:
    def __init__(self, lang='tr'):
        # OCR Engine optimized for CPU speed on macOS
        self.ocr = PaddleOCR(
            use_angle_cls=True, 
            lang='tr',
            det_db_thresh=0.1,
            det_db_box_thresh=0.3,
            det_db_unclip_ratio=2.0,
            text_det_limit_side_len=960,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False
        )
        self.preprocessor = ImagePreprocessor()
        
        # Initialize Structure Engine safely
        self.structure_engine = None
        if PPStructure is not None:
            try:
                self.structure_engine = PPStructure()
                logger.info("Table analysis engine (PPStructureV3) initialized.")
            except Exception as e:
                logger.warning(f"Failed to load PPStructure: {str(e)}")

    def _extract_blocks_from_result(self, result) -> list:
        """
        Robust parser supporting both traditional list and newest dictionary formats.
        """
        blocks = []
        if not result or not isinstance(result, list):
            return blocks

        for item in result:
            # Case 1: New Dictionary Format (Paddlex legacy)
            if isinstance(item, dict):
                polys = item.get("dt_polys", []) or []
                texts = item.get("rec_texts", []) or []
                scores = item.get("rec_scores", []) or []
                
                # Check for detection success but recognition failure
                if polys and not texts:
                    logger.warning("OCR Pass: Polygons detected but recognition returned no text.")
                
                count = max(len(polys), len(texts), len(scores))
                for i in range(count):
                    poly = polys[i] if i < len(polys) else None
                    text = texts[i] if i < len(texts) else ""
                    score = scores[i] if i < len(scores) else 0.0
                    if text and str(text).strip():
                        blocks.append([poly, [str(text).strip(), float(score)]])
            
            # Case 2: Traditional List Format [[box, [text, conf]], ...]
            elif isinstance(item, list):
                # Handle nested list-per-page [[[[box], [text, conf]]]]
                target = item[0] if len(item) > 0 and isinstance(item[0], list) and len(item[0]) > 0 and isinstance(item[0][0], list) else item
                for line in target:
                    if isinstance(line, list) and len(line) == 2 and isinstance(line[1], (list, tuple)):
                        blocks.append(line)
        
        return blocks

    def _upscale_if_needed(self, img):
        """
        Dynamically upscales low-resolution images. Capped at 1500px.
        """
        h, w = img.shape[:2]
        max_dim = 1200 # Lowered slightly for performance
        if max(w, h) < max_dim:
            scale = max_dim / max(w, h)
            new_w, new_h = int(w * scale), int(h * scale)
            return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        return img

    def _ensure_3_channel(self, img):
        if len(img.shape) == 2:
            return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        return img

    def _format_table_markdown(self, table_info: dict) -> str:
        html = table_info.get("res", {}).get("html", "")
        return f"HTML_TABLE:\n{html}\n" if html else ""

    def process(self, img_paths: list) -> dict:
        """
        High-Performance OCR with Hybrid Parsing and 15s Latency Gate.
        """
        import time
        if isinstance(img_paths, str): img_paths = [img_paths]
            
        final_report = {
            "all_ocr_results": [],
            "all_table_markdown": [],
            "metrics": {
                "total_blocks": 0, "total_chars": 0, "page_count": len(img_paths),
                "page_level": [], "pass_results": [],
                "confidence_variance": 0.0, "block_variance": 0.0, "total_ocr_duration": 0.0
            }
        }
        
        debug_mode = os.getenv("DEBUG_OCR", "False").lower() == "true"
        all_page_blocks = []
        all_page_confs = []
        global_start = time.perf_counter()
        
        for i, img_path in enumerate(img_paths):
            try:
                if not os.path.exists(img_path): continue
                raw_img = cv2.imread(img_path)
                if raw_img is None: continue
                
                # 0. Fast Reject
                hopeless, reason = self.preprocessor.is_hopeless(raw_img)
                if hopeless:
                    final_report["metrics"]["page_level"].append({"page": i+1, "status": "FAILED", "reason": reason})
                    continue

                robust_img = self._upscale_if_needed(raw_img)
                processed_img = self.preprocessor.process_numpy(robust_img)
                
                # DEBUG PHASE: Temporarily running ONLY RAW to verify parser success
                passes = [
                    {"name": "raw", "img": self._ensure_3_channel(robust_img)}
                    # Fallbacks (standard, inverted, adaptive) are paused for parser validation
                ]
                
                best_page_result = None
                pages_attempted = []
                
                for p_idx, p in enumerate(passes):
                    p_start = time.perf_counter()
                    
                    # 1. INFERENCE
                    ocr_res = self.ocr.ocr(p["img"])
                    inf_duration = time.perf_counter() - p_start
                    
                    # 2. COMPACT DIAGNOSTIC (Debug level)
                    page0 = ocr_res[0] if ocr_res and isinstance(ocr_res, list) else None
                    if isinstance(page0, dict):
                        logger.debug(f"P{i+1} [{p['name']}]: Found keys={list(page0.keys())} | Polys={len(page0.get('dt_polys',[]) or [])}")
                    
                    if inf_duration > 15.0:
                        logger.warning(f"LATENCY ALERT: Pass '{p['name']}' took {inf_duration:.1f}s.")

                    # 3. ROBUST PARSING
                    valid_blocks = self._extract_blocks_from_result(ocr_res)
                    block_count = len(valid_blocks)
                    avg_conf = sum([b[1][1] for b in valid_blocks]) / block_count if block_count > 0 else 0
                    
                    pass_metric = {
                        "name": p["name"], "blocks": block_count, "confidence": round(avg_conf, 4), "duration": round(inf_duration, 2)
                    }
                    pages_attempted.append(pass_metric)
                    logger.info(f"Page {i+1}: Result: Blocks={block_count}, Conf={avg_conf:.2f} | Time: {inf_duration:.2f}s")
                    
                    # 4. SHORT-CIRCUIT: Relaxed to > 0 for parser verification
                    if block_count > 0:
                        best_page_result = (valid_blocks, p["name"])
                        break
                    
                    if not best_page_result or block_count > len(best_page_result[0]):
                        best_page_result = (valid_blocks, p["name"])

                # Process Final Winner
                if best_page_result and best_page_result[0]:
                    v_blocks, method = best_page_result
                    p_chars = 0
                    for line in v_blocks:
                        box, (text, score) = line
                        p_chars += len(text)
                        final_report["all_ocr_results"].append({
                            "text": text, "confidence": float(score), "bbox": box, "page": i+1, "method": method
                        })
                    
                    all_page_blocks.append(len(v_blocks))
                    all_page_confs.append(sum([line[1][1] for line in v_blocks]) / len(v_blocks))
                    
                    final_report["metrics"]["total_blocks"] += len(v_blocks)
                    final_report["metrics"]["total_chars"] += p_chars
                    final_report["metrics"]["page_level"].append({
                        "page": i+1, "blocks": len(v_blocks), "chars": p_chars, "method": method
                    })
                else:
                    logger.error(f"Page {i+1}: ZERO blocks after OCR.")
                    final_report["metrics"]["page_level"].append({"page": i+1, "status": "FAILED"})
                
                final_report["metrics"]["pass_results"].append({"page": i+1, "results": pages_attempted})
                
                # DEBUG Visuals
                if debug_mode:
                    for p in passes:
                        cv2.imwrite(f"{img_path}_p{i+1}_{p['name']}.png", p["img"])

                # 5. Structure Analysis (Only if text found)
                if self.structure_engine and best_page_result and best_page_result[0]:
                    winner_img = passes[0]["img"]
                    struct_res = self.structure_engine(winner_img)
                    for region in struct_res:
                        if region["type"] == "table":
                            md = self._format_table_markdown(region)
                            if md: final_report["all_table_markdown"].append(f"Page {i+1} Table:\n{md}")
                            
            except Exception as e:
                logger.exception(f"Processor Critical on page {i+1}: {str(e)}")
        
        # Final Metrics Calculation
        final_report["metrics"]["total_ocr_duration"] = round(time.perf_counter() - global_start, 2)
        if len(all_page_blocks) > 0:
            final_report["metrics"]["block_variance"] = float(np.var(all_page_blocks))
            final_report["metrics"]["confidence_variance"] = float(np.var(all_page_confs))
                
        return final_report
