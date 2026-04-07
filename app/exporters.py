import json
import pandas as pd
from lxml import etree
import uuid
from datetime import datetime

class DocumentExporter:
    """
    Handles data export to various formats: CSV, Excel, and UBL-TR XML.
    """
    
    @staticmethod
    def to_csv(extraction_data: dict, output_path: str):
        """Converts extraction JSON to a flat CSV."""
        hierarchy = extraction_data.get("financial_hierarchy", {})
        root = hierarchy.get("root_transaction", {})
        adjustments = hierarchy.get("adjustments_and_fees", [])
        
        # Flattening for CSV
        rows = []
        base_info = {
            "Filename": extraction_data.get("filename", "N/A"),
            "Total_Amount": root.get("amount", 0.0),
            "Label": root.get("label", ""),
            "Text_Confirmation": root.get("text_confirmation", ""),
            "Document_Type": extraction_data.get("document_analysis", {}).get("type", "UNKNOWN")
        }
        
        if not adjustments:
            rows.append(base_info)
        else:
            for adj in adjustments:
                row = base_info.copy()
                row.update({
                    "Adjustment_Group": adj.get("group_name", ""),
                    "Group_Total": adj.get("total_impact", 0.0),
                    "Math_Status": adj.get("math_status", ""),
                    "Breakdown": json.dumps(adj.get("breakdown", {}))
                })
                rows.append(row)
        
        df = pd.DataFrame(rows)
        df.to_csv(output_path, index=False, encoding='utf-8-sig')
        return output_path

    @staticmethod
    def to_ubl_tr(extraction_data: dict, output_path: str):
        """
        Generates a simplified UBL-TR 2.1 compliant XML.
        Warning: This is a structural template, real GIB compliance requires full schema filling.
        """
        hierarchy = extraction_data.get("financial_hierarchy", {})
        root = hierarchy.get("root_transaction", {})
        
        # XML Namespace Map
        NS_MAP = {
            None: "urn:oasis:names:specification:ubl:schema:xsd:Invoice-2",
            "cac": "urn:oasis:names:specification:ubl:schema:xsd:CommonAggregateComponents-2",
            "cbc": "urn:oasis:names:specification:ubl:schema:xsd:CommonBasicComponents-2"
        }
        
        invoice = etree.Element("Invoice", nsmap=NS_MAP)
        
        # Header Info
        etree.SubElement(invoice, "{%s}UBLVersionID" % NS_MAP["cbc"]).text = "2.1"
        etree.SubElement(invoice, "{%s}ID" % NS_MAP["cbc"]).text = str(uuid.uuid4())[:16].upper()
        etree.SubElement(invoice, "{%s}IssueDate" % NS_MAP["cbc"]).text = datetime.now().strftime("%Y-%m-%d")
        
        # Monetary Total
        legal_total = etree.SubElement(invoice, "{%s}LegalMonetaryTotal" % NS_MAP["cac"])
        payable_amount = etree.SubElement(legal_total, "{%s}PayableAmount" % NS_MAP["cbc"], currencyID="TRY")
        payable_amount.text = str(root.get("amount", 0.0))
        
        # Serialize
        tree = etree.ElementTree(invoice)
        tree.write(output_path, pretty_print=True, xml_declaration=True, encoding="UTF-8")
        return output_path
