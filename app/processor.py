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
        # OCR Engine for standard text with high-sensitivity parameters
        self.ocr = PaddleOCR(
            use_angle_cls=True, 
            lang='tr',
            det_db_thresh=0.1,        # Ultra-sensitive threshold
            det_db_box_thresh=0.3,    # Lower box threshold
            det_db_unclip_ratio=2.0,  # Expand boxes to catch artifacts
            text_det_limit_side_len=1500 # Updated non-deprecated parameter
        )
        self.preprocessor = ImagePreprocessor()
        
        # Initialize Structure Engine safely
        self.structure_engine = None
        if PPStructure is not None:
            try:
                # Basic initialization - avoiding show_log conflicts
                self.structure_engine = PPStructure()
                logger.info("Table analysis engine (PPStructureV3) initialized.")
            except Exception as e:
                logger.warning(f"Failed to load PPStructure: {str(e)}. Table analysis skipped.")
                self.structure_engine = None
        else:
            logger.warning("PPStructure module not available. Layout analysis will be limited.")

    def _upscale_if_needed(self, img):
        """
        Dynamically upscales low-resolution images to help OCR detection.
        Capped at 1500px for performance on CPU.
        """
        h, w = img.shape[:2]
        max_dim = 1500
        if max(w, h) < max_dim:
            scale = max_dim / max(w, h)
            # Limit scale to avoid memory issues
            scale = min(scale, 2.0)
            new_w = int(w * scale)
            new_h = int(h * scale)
            logger.info(f"Upscaling image for OCR: {w}x{h} -> {new_w}x{new_h} (Scale: {scale:.2f}x)")
            return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        return img

    def _ensure_3_channel(self, img):
        """Ensures image is 3-channel BGR for PaddleX engine compatibility."""
        if len(img.shape) == 2:
            return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        return img

    def _format_table_markdown(self, table_info: dict) -> str:
        """
        Converts PaddleStructure table info to Markdown format.
        """
        html = table_info.get("res", {}).get("html", "")
        if not html: return ""
        return f"HTML_TABLE:\n{html}\n"

    def process(self, img_paths: list) -> dict:
        """
        Raw-First OCR Pipeline with Clean Doc Bypass and Tiered Fallback.
        """
        import time
        if isinstance(img_paths, str):
            img_paths = [img_paths]
            
        final_report = {
            "all_ocr_results": [],
            "all_table_markdown": [],
            "metrics": {
                "total_blocks": 0,
                "total_chars": 0,
                "page_count": len(img_paths),
                "page_level": [],
                "pass_results": [],
                "confidence_variance": 0.0,
                "block_variance": 0.0,
                "total_ocr_duration": 0.0
            }
        }
        
        debug_mode = os.getenv("DEBUG_OCR", "False").lower() == "true"
        all_page_blocks = []
        all_page_confs = []
        global_start = time.perf_counter()
        
        for i, img_path in enumerate(img_paths):
            try:
                if not os.path.exists(img_path): continue
                
                # 1. Load and Fast-Fail Check
                raw_img = cv2.imread(img_path)
                if raw_img is None: continue
                
                hopeless, reason = self.preprocessor.is_hopeless(raw_img)
                if hopeless:
                    logger.warning(f"Page {i+1}: Fast-fail triggered. Reason: {reason}")
                    final_report["metrics"]["page_level"].append({
                        "page": i+1, "status": "FAILED", "fast_fail_reason": reason
                    })
                    continue

                # 2. Preparation (Upscale + Heuristics)
                prep_start = time.perf_counter()
                robust_img = self._upscale_if_needed(raw_img)
                is_clean = self.preprocessor.is_clean_document(robust_img)
                if is_clean:
                    logger.info(f"Page {i+1}: Clean document detected. Favoring raw pass.")
                
                processed_img = self.preprocessor.process_numpy(robust_img)
                prep_duration = time.perf_counter() - prep_start
                
                # Tiered Execution Suite (Re-ordered to prioritize RAW)
                passes = [
                    {"name": "raw", "img": self._ensure_3_channel(robust_img)},
                    {"name": "standard", "img": self._ensure_3_channel(processed_img)},
                    {"name": "inverted", "img": self._ensure_3_channel(cv2.bitwise_not(processed_img))},
                    {"name": "adaptive", "img": self._ensure_3_channel(self.preprocessor.adaptive_threshold(processed_img))}
                ]
                
                best_page_result = None
                pages_attempted = []
                
                for p_idx, p in enumerate(passes):
                    p_start = time.perf_counter()
                    
                    # LOG VISUAL STATS
                    img_stats = f"Mean: {np.mean(p['img']):.1f}, Std: {np.std(p['img']):.1f}"
                    logger.info(f"Page {i+1}: Pass '{p['name']}' started | {img_stats}")
                    
                    # OCR Inference (In-Memory Numpy) with 15s Latency Check
                    inf_start = time.perf_counter()
                    ocr_res = self.ocr.ocr(p["img"])
                    inf_duration = time.perf_counter() - inf_start
                    
                    # 1. MANDATORY DIAGNOSTIC LOGGING
                    logger.info(f"Page {i+1} [{p['name']}]: OCR RAW OUTPUT TYPE: {type(ocr_res)}")
                    logger.info(f"Page {i+1} [{p['name']}]: OCR RAW OUTPUT: {repr(ocr_res)[:1500]}")
                    
                    if inf_duration > 15.0:
                        logger.warning(f"CRITICAL LATENCY: Pass '{p['name']}' took {inf_duration:.2f}s (>15s limit). Aborting pass.")
                        pages_attempted.append({"name": p["name"], "status": "ABORTED_TOO_SLOW", "duration": round(inf_duration, 2)})
                        continue
                    
                    # 2. ROBUST PARSING (Handle varied nesting levels and Paddlex objects)
                    parse_start = time.perf_counter()
                    valid_blocks = []
                    
                    try:
                        # Attempt to normalize common PaddleOCR formats
                        if ocr_res and isinstance(ocr_res, list):
                            # Case A: Standard result [[[box], [text, conf]], ...]
                            # or nested list-per-page [[[[box], [text, conf]]]]
                            target = ocr_res[0] if len(ocr_res) > 0 and isinstance(ocr_res[0], list) and len(ocr_res[0]) > 0 and isinstance(ocr_res[0][0], list) else ocr_res
                            
                            # Deep search for line blocks
                            for line in target:
                                # We expect [bbox, [text, conf]]
                                if isinstance(line, list) and len(line) == 2 and isinstance(line[1], (list, tuple)):
                                    valid_blocks.append(line)
                                elif hasattr(line, 'to_dict'): # Paddlex object fallback
                                    d = line.to_dict()
                                    # Translate to standard format if possible
                                    pass 
                    except Exception as parse_err:
                        logger.error(f"Parsing error on pass {p['name']}: {str(parse_err)}")

                    parse_duration = time.perf_counter() - parse_start
                    block_count = len(valid_blocks)
                    page_scores = [line[1][1] for line in valid_blocks]
                    avg_conf = sum(page_scores) / block_count if block_count > 0 else 0
                    
                    pass_metric = {
                        "name": p["name"], 
                        "blocks": block_count, 
                        "confidence": round(avg_conf, 4), 
                        "duration": round(inf_duration + parse_duration, 4),
                        "debug": {
                            "prep": round(prep_duration, 4),
                            "inference": round(inf_duration, 4),
                            "parse": round(parse_duration, 4)
                        }
                    }
                    pages_attempted.append(pass_metric)
                    logger.info(f"Page {i+1}: Result: Blocks={block_count}, Conf={avg_conf:.2f} (Inf: {inf_duration:.2f}s, Parse: {parse_duration:.4f}s)")
                    
                    # SHORT-CIRCUIT
                    if block_count > 5 and avg_conf > 0.8:
                        logger.info(f"Page {i+1}: Short-circuit on '{p['name']}'")
                        best_page_result = (valid_blocks, p["name"])
                        break
                    
                    if not best_page_result or block_count > len(best_page_result[0]):
                        best_page_result = (valid_blocks, p["name"])

                # Process Final Winner for this Page
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
                    logger.error(f"Page {i+1}: FAILED after {len(pages_attempted)} passes.")
                    final_report["metrics"]["page_level"].append({"page": i+1, "status": "FAILED"})
                
                final_report["metrics"]["pass_results"].append({"page": i+1, "results": pages_attempted})
                
                # DEBUG Visuals
                if debug_mode:
                    for p in passes:
                        debug_path = f"{img_path}_p{i+1}_{p['name']}.png"
                        cv2.imwrite(debug_path, p["img"])

                # 3. Structure Analysis (V3)
                if self.structure_engine and best_page_result and best_page_result[0]:
                    struct_start = time.perf_counter()
                    # Feed the successful image to the layout engine
                    # If Pass 0 (raw) was used, we use Pass 0 image
                    winner_img = passes[0]["img"] 
                    for p in passes:
                        if p["name"] == best_page_result[1]:
                            winner_img = p["img"]
                            break
                            
                    struct_res = self.structure_engine(winner_img)
                    for region in struct_res:
                        if region["type"] == "table":
                            md = self._format_table_markdown(region)
                            if md: final_report["all_table_markdown"].append(f"Page {i+1} Table:\n{md}")
                    logger.info(f"Page {i+1}: Layout Analysis completed in {time.perf_counter() - struct_start:.2f}s")
                            
            except Exception as e:
                logger.exception(f"Processor Critical on page {i+1}: {str(e)}")
        
        # Final Metrics
        final_report["metrics"]["total_ocr_duration"] = round(time.perf_counter() - global_start, 2)
        if len(all_page_blocks) > 1:
            final_report["metrics"]["block_variance"] = float(np.var(all_page_blocks))
            final_report["metrics"]["confidence_variance"] = float(np.var(all_page_confs))
                
        return final_report
